#!/usr/bin/env python
# -*- coding: utf-8 -*-

import models.drn as drn
from models.DRNSeg import DRNSeg
from models.FCN32s import FCN32s
import data_transforms as transforms
import json
import math
import os
from os.path import exists, join, split
import threading

import time, datetime

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from learning.utils_learn import *
from learning.dataloader import SegList, SegListMS, get_info, get_loader
import logging
from learning.validate import validate
import data_transforms as transforms

from dataloaders.utils import decode_segmap
from torch.utils.tensorboard import SummaryWriter
from learning.attack import PGD_attack


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


log_epsilon = 1e-20
epsilon = 1e-10

def ensemble_entropy(y_pred):
    num_pred = y_pred.size(2) * y_pred.size(3)
    entropy_type = "sum_entropy"
    if entropy_type == "all_entropy":
        flag_pred = y_pred.view(y_pred.size(0), -1)
        entropy = torch.sum(-flag_pred * torch.log(flag_pred + log_epsilon)) #TODO: here, even sum the batch dim
    elif entropy_type == "sum_entropy":  # Pang et al
        sum_score = torch.sum(torch.sum(y_pred, dim=3), dim=2) / num_pred
        entropy = torch.sum(-sum_score * torch.log(sum_score + log_epsilon))
        # print("\n_____Debugging entropy", "y_pred.shape", y_pred.shape,
        #   # "sum", torch.sum(y_pred, axis=3).shape,
        #   # "sum of sum", torch.sum(torch.sum(y_pred, axis=3), axis=2).shape,
        #   "num_pred", num_pred,
        #   "sum_score", sum_score,
        #   "entropy", entropy, torch.log(sum_score + log_epsilon),"\n",
        #   # "individual elements", np.where(y_pred.detach().numpy() < 0)[0].shape,
        #       "\n\n")
    elif entropy_type == "mutual_info": # borrow the mutual information idea, where it is the total entropy - mean of individual entropy
        pass
    return entropy


# def log_det(y_true, y_pred, num_model=FLAGS.num_models):
#     bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
#     mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
#     mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
#     mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
#     matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
#     all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
#     return all_log_det

#TODO: also a function with global logdet:

def log_det_global_cuda():
    pass

#TODO: this is a local logdet loss
def log_det_cuda(y_pred, y_class_true, args, neglect = 255):  # We need to max Diversity for each class

    # TODO: we should first down sampling before move on (like dropout 99%)
    delta_det = 1e-3

    drop_ratio = 1 - args.drop_ratio

    # y_true need to be one hot
    # print('class num', y_pred.size(1))
    # mark_neglect = torch.ones_like(y_class_true) * neglect == y_class_true
    if torch.cuda.is_available():
        mark_neglect = torch.cuda.LongTensor(y_class_true.size(0), y_class_true.size(1), y_class_true.size(2)).fill_(1) * neglect == y_class_true
    else:
        mark_neglect = torch.LongTensor(y_class_true.size(0), y_class_true.size(1), y_class_true.size(2)).fill_(
            1) * neglect == y_class_true
    # print("mask", torch.max(mark_neglect))

    y_class_true = y_class_true * (1 - mark_neglect.long())
    # print('max 18? Yes, it is , starting from 0 then 18', y_class_true.max())

    y_class_true = y_class_true * (1 - mark_neglect.long()) + (y_pred.size(1)) * mark_neglect.long()  # we put 18 + 1 as the neglect class
    # print('lab', torch.max(y_class_true))
    # print(y_pred.size(1)+1)
    if torch.cuda.is_available():
        y_class_true = y_class_true.cuda()
    y_true = torch.nn.functional.one_hot(y_class_true, y_pred.size(1)+1)

    # print('one hot size', y_true.size())
    y_true = y_true[:,:,:,:y_pred.size(1)]
    # print('one hot size', y_true.size(), 'pred size', y_pred.size())
    if torch.cuda.is_available():
        non_max_mask = (torch.ones_like(y_true).cuda() - y_true) != torch.zeros_like(y_true).cuda()
    else:
        non_max_mask = (torch.ones_like(y_true) - y_true) != torch.zeros_like(y_true)

    non_max_mask = torch.transpose(non_max_mask, dim0=1, dim1=3).float()

    # print('non_max_mask', non_max_mask.size())
    # print("HERE", non_max_mask.shape, y_pred.shape)
    mask_non_y_pred = non_max_mask * y_pred
    mask_non_y_pred = mask_non_y_pred.view(-1, mask_non_y_pred.size(1), mask_non_y_pred.size(2) * mask_non_y_pred.size(3))
    mask_non_y_pred = torch.transpose(mask_non_y_pred, dim0=1, dim1=2)  # batch * num_pixel * class_num

    #TODO: now , we look at the diversity within the groundtruth class
    # class_categories = set(y_class_true.cpu().numpy())
    det_loss = 0
    together = False

    if together:
        if torch.cuda.is_available():
            drop_approximate = torch.cuda.FloatTensor(mask_non_y_pred.size(0), mask_non_y_pred.size(1), 1).uniform_() > drop_ratio
        else:
            drop_approximate = torch.FloatTensor(mask_non_y_pred.size(0), mask_non_y_pred.size(1), 1).uniform_() > drop_ratio

        drop_approximate = drop_approximate.repeat(1, 1, mask_non_y_pred.size(2))

        select_element = mask_non_y_pred[drop_approximate]

        element_same_class = select_element.view(-1, y_pred.size(1))
        element_same_class = element_same_class / ((torch.sum(element_same_class ** 2, dim=1).unsqueeze(
            1)+ epsilon) ** 0.5)  # pixel_num * fea_len   TODO: need epsilon or result in NAN

        matrix = torch.mm(element_same_class,
                          torch.transpose(element_same_class, dim0=0, dim1=1))  # batch * pixel_num * pixel_num
        if torch.cuda.is_available():  # TODO: can be written in a shorter way
            logdet_loss = torch.logdet(matrix[0] + delta_det * torch.eye(matrix.size(1)).cuda())
        else:
            logdet_loss = torch.logdet(matrix[0] + delta_det * torch.eye(matrix.size(1)))

        ## print("logdet this", logdet_loss)
        det_loss = logdet_loss

    else:
        for each_category in range(y_pred.size(1)):
            # ind_select = torch.ones_like(y_class_true).cuda() * each_category == y_class_true
            if torch.cuda.is_available():
                ind_select = torch.cuda.LongTensor(y_class_true.size(0), y_class_true.size(1), y_class_true.size(2)).fill_(1) * each_category == y_class_true
            else:
                ind_select = torch.LongTensor(y_class_true.size(0), y_class_true.size(1), y_class_true.size(2)).fill_(1) * each_category == y_class_true

            # print(ind_select.size())
            if torch.cuda.is_available():
                drop_approximate = torch.cuda.FloatTensor(ind_select.size()).uniform_() > drop_ratio
            else:
                drop_approximate = torch.FloatTensor(ind_select.size()).uniform_() > drop_ratio

            ind_select = ind_select * drop_approximate

            # If the class exist in the input
            if torch.sum(ind_select) > 0:
                # print('each cate', each_category)

                flat_ind_select = ind_select.view(-1, ind_select.size(1)*ind_select.size(2)) # batch * num_pixel
                flat_ind_select = flat_ind_select.unsqueeze(2) # batch * num_pixel * 1
                flat_ind_select = flat_ind_select.repeat(1, 1, mask_non_y_pred.size(2))

                batch_wise = True
                if batch_wise:
                    # iterating over each batch # can be replace with a batch global one, will check the running time
                    for batch_i in range(flat_ind_select.size(0)):
                        mask_non_y_pred_b = mask_non_y_pred[batch_i]
                        flat_ind_select_b = flat_ind_select[batch_i]

                        if torch.sum(flat_ind_select_b)==0:
                            continue

                        element_same_class = mask_non_y_pred_b[flat_ind_select_b]  # selecting the predict score only for that "category" class;
                        # we expect the length shrink
                        # size: batch * num_pixel * feature_length

                        # TODO: check how to reshape this back !I have checked, this is correct.
                        element_same_class = element_same_class.view(-1, y_pred.size(1))
                        # print('after reshape', element_same_class)

                        # print("\nELEMENT_SAME_CLASS2 ","#blank elem",np.isnan(element_same_class.clone().detach().numpy()).sum(),"#0s ",len(np.where(element_same_class.clone().detach().numpy() ==0.)[0]),np.max(element_same_class.clone().detach().numpy()), np.min(element_same_class.clone().detach().numpy()))

                        # TODO: Normalize the score feature vector for each pixel
                        #TODO: it is crucial to add epsilon inside the root operation follows in order to prevent NAN during BP
                        # So I think norm should be preferred

                        element_same_class = element_same_class / (((torch.sum(element_same_class ** 2, dim=1).unsqueeze(1))+ epsilon) ** 0.5 + epsilon) # pixel_num * fea_len
                        # element_same_class = element_same_class / torch.norm(element_same_class, dim=2, keepdim=True)

                        matrix = torch.mm(element_same_class, torch.transpose(element_same_class, dim0=0, dim1=1)) # batch * pixel_num * pixel_num

                        if torch.cuda.is_available(): #TODO: can be written in a shorter way
                            logdet_loss = torch.logdet(matrix[0] + delta_det * torch.eye(matrix.size(1)).cuda())
                        else:
                            logdet_loss = torch.logdet(matrix[0] + delta_det * torch.eye(matrix.size(1)))

                        det_loss += logdet_loss
                else:
                    element_same_class = mask_non_y_pred[flat_ind_select]
                    #TODO: check how to reshape this back !!!!!!!!!
                    element_same_class = element_same_class.view(-1, y_pred.size(1))
                    element_same_class = element_same_class / (torch.norm(element_same_class, dim=1, keepdim=True) + epsilon) #TODO: divided , you need epsilon to prevent NAN

                    matrix = torch.mm(element_same_class,
                                       torch.transpose(element_same_class, dim0=0, dim1=1))  # pixel_num * pixel_num
                    #
                    logdet_loss = torch.logdet(matrix + delta_det * torch.eye(matrix.size(1).cuda()))
                    det_loss += logdet_loss

    return det_loss






#TODO: Can we use GAN to align the variance? Use Pang et al? Maximum the overall entropy?

def train_seg_reg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    # print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained and args.loading:
        print('args.pretrained', args.pretrained)
        single_model.load_state_dict(torch.load(args.pretrained))

    out_dir = 'output/{}_{:03d}_{}'.format(args.arch, 0, args.phase)


    model = torch.nn.DataParallel(single_model)
    criterion = nn.NLLLoss(ignore_index=255)


    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()


    # Data loading code
    info = get_info(args.dataset)
    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val", out_name=True)
    adv_val_loader = get_loader(args, "adv_val", out_name=True)


    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # Backup files before resuming/starting training
    backup_output_dir = args.backup_output_dir
    os.makedirs(backup_output_dir, exist_ok=True)
    if os.path.exists(backup_output_dir):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_backup_folder = "reg_adv_train_" + args.arch + "_" + args.dataset + "_" + timestamp
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_backup_folder)
        print(experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder, ignore=include_patterns('*.py', '*.json'))


    # Logging with TensorBoard
    log_dir = os.path.join(experiment_backup_folder, "runs")
    val_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')

    writer = SummaryWriter(log_dir=log_dir)
    fh = logging.FileHandler(experiment_backup_folder + '/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # optionally resume from a checkpoint
    if args.resume:
        print("resuming", args.resume)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:

        validate(val_loader, model, criterion,args=args,log_dir=experiment_backup_folder, eval_score=accuracy, info=info)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        reg_train(train_loader, model, criterion, optimizer, epoch, args, info, writer, args.dataset,
              eval_score=accuracy)

        # evaluate on validation set
        prec = validate(val_loader, model, criterion, args=args,log_dir=experiment_backup_folder, eval_score=accuracy,
                        info=info, writer=val_writer, epoch=epoch)
        if epoch % args.val_freq:
            from learning.validate import validate_adv
            mAP = validate_adv(adv_val_loader, model, args.classes, save_vis=True,
                               has_gt=True, output_dir=out_dir, downsize_scale=args.downsize_scale,
                               args=args, info=info, writer=val_writer, epoch=epoch)
            logger.info('adv mAP: %f', mAP)
            # writer.add_scalar('Reg_Adv_Validate/prec', prec, epoch)
            writer.add_scalar('Reg_Adv_Validate/mAP', mAP, epoch)

        is_best = prec > best_prec1
        if is_best:
            best_prec1 = max(mAP, best_prec1)

            save_model_path = os.path.join(experiment_backup_folder, 'savecheckpoint')
            os.makedirs(save_model_path, exist_ok=True)
            checkpoint_path = os.path.join(save_model_path, 'checkpoint_latest.pth.tar')

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path, save_model_path=save_model_path)
            if (epoch + 1) % 1 == 0:
                # history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
                shutil.copyfile(checkpoint_path, history_path)
    writer.close()

def reg_train(train_loader, model, criterion, optimizer, epoch, args, info, writer, dataset,
          eval_score=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    entropy_losses = AverageMeter()
    classify_losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print("Standard Training + Regularization" if not args.adv_train else "Adversarial Training + Regularization")

    for i, (input, target) in enumerate(train_loader):
        # print('target', target)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.adv_train:
            adv_img = PGD_attack(input, target, model, criterion, args.epsilon, args.steps, args.dataset,
                                 args.step_size, info, using_noise=True)
        else:
            adv_img = input

        # input = input.cuda()
        # print('diff', (adv_img.data-input) / (args.epsilon))

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        # TODO: adversarial training
        clean_input = input
        input = adv_img.data
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            clean_input = clean_input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target, requires_grad=False)
        clean_input_var = torch.autograd.Variable(clean_input)

        # compute output
        # output = model(input_var)[0]
        output, _, softmax_output = model(input_var)
        cla_loss = criterion(output, target_var)

        # TODO: we are random sampling this, we may want to do several times

        reg_loss_total = 0
        for iii in range(args.MC_times):
            reg_loss = log_det_cuda(softmax_output, target_var, args)   #y_true, y_pred, y_class_true
            reg_loss_total += reg_loss

        reg_term = args.reg_lambda / args.MC_times * reg_loss_total

        # entropy_loss = reg_term
        # entropy_loss = 0
        entropy_loss = args.entropy_lambda * ensemble_entropy(softmax_output)

        loss = cla_loss - reg_term - entropy_loss

        losses.update(loss.data.item(), input.size(0))
        classify_losses.update(cla_loss.data.item(), input.size(0))
        reg_losses.update(reg_term.data.item(), input.size(0))
        entropy_losses.update(entropy_loss.data.item(), input.size(0))

        if eval_score is not None:
            if target_var.size(0)>0:
                scores.update(eval_score(output, target_var), input.size(0))
            else:
                print("0 size!")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.debug):
            print_freq = 10

        if i % (args.print_freq // args.batch_size) == 0:
            # Convert target and prediction to rgb images to visualise
            class_prediction = torch.argmax(output, dim=1)
            decoded_target = decode_segmap(target[0].cpu().numpy() if torch.cuda.is_available() else target[0].numpy(),
                                           dataset)
            decoded_target = np.moveaxis(decoded_target, 2, 0)
            decoded_class_prediction = decode_segmap(
                class_prediction[0].cpu().numpy() if torch.cuda.is_available() else class_prediction[0].numpy(),
                dataset)
            decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)


            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Xent Loss {classify_losses.val:.4f} ({classify_losses.avg:.4f})\t'
                        'Reg Loss {reg_losses.val:.4f} ({reg_losses.avg:.4f})\t'
                        'Entropy Loss {entro_losses.val:.4f} ({entro_losses.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, classify_losses=classify_losses, reg_losses=reg_losses,
                entro_losses = entropy_losses, loss=losses, top1=scores))

            # compute output for the case when clean image is passed.
            clean_output = model(clean_input_var)[0]
            clean_loss = criterion(clean_output, target_var)
            if eval_score is not None:
                if target_var.size(0) > 0:
                    clean_score = eval_score(clean_output, target_var)
                else:
                    print("0 size!")
                    clean_score = 0

            writer.add_image('Image/adv image ', back_transform(input_var, info)[0])
            writer.add_image('Image/clean image ', back_transform(clean_input_var, info)[0])
            writer.add_image('Image/image target ', decoded_target)
            writer.add_image('Image/image prediction ', decoded_class_prediction)

            writer.add_scalar('Reg_Adv_Train/Score', scores.val, epoch * len(train_loader) + i)
            writer.add_scalar('Reg_Adv_Train/Loss', losses.val, epoch * len(train_loader) + i)
            writer.add_scalar('Reg_Adv_Train/Clean_Score', clean_score, epoch * len(train_loader) + i)
            writer.add_scalar('Reg_Adv_Train/Clean_Loss', clean_loss, epoch * len(train_loader) + i)

            writer.add_scalar('Reg_Adv_Train/Classify_Loss', classify_losses.val, epoch * len(train_loader) + i)
            writer.add_scalar('Reg_Adv_Train/Reg_Loss', reg_losses.val, epoch * len(train_loader) + i)

            if args.debug and i==(args.print_freq // args.batch_size)*10: #breaking after 10 images.
                break

        # break


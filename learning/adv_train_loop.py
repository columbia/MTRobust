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
from learning.dataloader import SegList, SegListMS, get_loader, get_info
import logging
from learning.validate import validate
import data_transforms as transforms

import matplotlib.pyplot as plt

from learning.attack import PGD_attack
from dataloaders.utils import decode_segmap
from torch.utils.tensorboard import SummaryWriter

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_seg_adv(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

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
    if os.path.exists(backup_output_dir):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_backup_folder = "adv_train_" + args.arch + "_" + args.dataset + "_" + timestamp
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_backup_folder)
        print(experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder, ignore=include_patterns('*.py', '*.json'))
    else:
        experiment_backup_folder = ""
        print("backup_output_dir does not exist")

    #Logging with TensorBoard
    log_dir = experiment_backup_folder+"/runs/"
    writer = SummaryWriter(log_dir=log_dir)
    val_writer = SummaryWriter(log_dir=log_dir+'/validate_runs/')

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

        validate(val_loader, model, criterion,args=args, log_dir=experiment_backup_folder, eval_score=accuracy,info=info)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        adv_train(train_loader, model, criterion, optimizer, epoch, args, info, writer, args.dataset,
              eval_score=accuracy)
        # evaluate on validation set

        #TODO: definitely uncomment this.
        prec = validate(val_loader, model, criterion, args=args, log_dir=experiment_backup_folder,
                        eval_score=accuracy, info=info, epoch=epoch, writer=val_writer) #To see the accuracy on clean images as well.
        from learning.validate import validate_adv
        mAP = validate_adv(adv_val_loader, model, args.classes, save_vis=True, log_dir=experiment_backup_folder,
                           has_gt=True, output_dir=out_dir, downsize_scale=args.downsize_scale,
                           args=args, info=info, writer=val_writer, epoch=epoch)
        logger.info('adv mAP: %f', mAP)
        # writer.add_scalar('Adv_Validate/prec', prec, epoch)
        # writer.add_scalar('Adv_Validate/mAP', mAP, epoch)

        is_best = mAP > best_prec1
        if is_best:
            best_prec1 = max(mAP, best_prec1)

            # checkpoint_path = 'checkpoint_latest.pth.tar'
            save_model_path = os.path.join(experiment_backup_folder, 'savecheckpoint')
            os.makedirs(save_model_path, exist_ok=True)
            checkpoint_path = os.path.join(save_model_path, 'checkpoint_latest.pth.tar')

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path,save_model_path = save_model_path)
            if (epoch + 1) % 10 == 0:
                # history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
                history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
                shutil.copyfile(checkpoint_path, history_path)

    writer.close()


def adv_train(train_loader, model, criterion, optimizer, epoch, args, info, writer, dataset,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #Get attacked image
        adv_img = PGD_attack(input, target, model, criterion, args.epsilon, args.steps, args.dataset,
                             args.step_size, info, using_noise=True)



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
            target = target.cuda()
            clean_input = clean_input.cuda()
        input_var = torch.autograd.Variable(input)
        clean_input_var = torch.autograd.Variable(clean_input)
        target_var = torch.autograd.Variable(target)

        if args.debug:
            # Debug visualisations
            print("\nVisualising here.\n",
                  # Add things to print here
                  )
            # print("Max in difference is ", np.max(diff_img))
            arr_clean_img = clean_input_var.clone().cpu().numpy() if torch.cuda.is_available() else clean_input_var.clone().numpy()
            arr_adv_img = input_var.clone().cpu().numpy() if torch.cuda.is_available() else input_var.clone().numpy()
            vis_clean_img = np.moveaxis(np.copy(arr_clean_img)[0], 0, 2)
            vis_adv_img = np.moveaxis(np.copy(arr_adv_img)[0], 0, 2)
            vis_clean_img_tr = np.moveaxis(back_transform(np.copy(arr_clean_img), info)[0],0,2) # np.copy is necessary to make sure that the original arrays are not modified.
            vis_adv_img_tr = np.moveaxis(back_transform(np.copy(arr_adv_img), info)[0],0,2) # np.copy is necessary to make sure that the original arrays are not modified.
            diff_img1 = vis_adv_img - vis_clean_img
            diff_img2 = vis_adv_img_tr - vis_clean_img_tr
            f, axarr = plt.subplots(1, 6)
            axarr[0].imshow(vis_clean_img)
            axarr[1].imshow(vis_adv_img)
            axarr[2].imshow(vis_clean_img_tr)
            axarr[3].imshow(vis_adv_img_tr)
            # print()
            axarr[4].imshow(diff_img1)
            axarr[5].imshow(diff_img2)
            plt.show()
            print("__max diff is__",np.max(diff_img1)*255,np.max(diff_img2)*255)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        losses.update(loss.data.item(), input.size(0))
        if eval_score is not None:
            if target_var.size(0)>0:
                scores.update(eval_score(output, target_var), input.size(0))
            else:
                print("0 size!")
                clean_score = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.debug:
            print_freq = 10

        if i % print_freq == 0:
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
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))
            #TODO: Get training accuracy for non adv images. (forward the clean data.)

            # compute output for the case when clean image is passed.
            clean_output = model(clean_input_var)[0]
            clean_loss = criterion(clean_output, target_var)
            if eval_score is not None:
                if target_var.size(0) > 0:
                    clean_score = eval_score(clean_output, target_var)
                else:
                    print("0 size!")
                    clean_score = 0

            # print('img max m,in', torch.max(input_var[0]), torch.min(input_var[0]))
            # print('img max m,in', torch.max(clean_input_var[0]), torch.min(clean_input_var[0]))
            # input_var[0] = (back_transform(input_var[0], info) * 255).long()

            writer.add_image('Image/adv image ', back_transform(input_var, info)[0])
            writer.add_image('Image/clean image ', back_transform(clean_input_var, info)[0])
            writer.add_image('Image/image target ', decoded_target)
            writer.add_image('Image/image prediction ', decoded_class_prediction)
            writer.add_scalar('Adv_Train/Score', scores.val, epoch*len(train_loader)+i)
            writer.add_scalar('Adv_Train/Loss', losses.val, epoch*len(train_loader)+i)
            writer.add_scalar('Adv_Train/Clean_Score', clean_score, epoch*len(train_loader)+i)
            writer.add_scalar('Adv_Train/Clean_Loss', clean_loss, epoch*len(train_loader)+i)

            if args.debug and i == print_freq:
                break



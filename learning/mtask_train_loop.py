#!/usr/bin/env python
# -*- coding: utf-8 -*-

from comet_ml import Experiment

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

from dataloaders.utils import decode_segmap

from torch.utils.tensorboard import SummaryWriter
from learning.mtask_validate import mtask_validate
import torchvision


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_mtasks(args):
    # batch_size = args.batch_size
    # num_workers = args.workers
    #

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    from models.mtask_losses import get_losses_and_tasks

    criteria, taskonomy_tasks = get_losses_and_tasks(args)

    if args.arch == 'resnet-18':
        from models.taskonomy_models import resnet18_taskonomy
        model = resnet18_taskonomy(pretrained=False, tasks=args.task_set)

    elif args.arch == 'resnet-50':
        from models.taskonomy_models import resnet50_taskonomy
        model = resnet50_taskonomy(pretrained=False, tasks=args.task_set)

    elif 'drn' in args.arch:
        # CHANGE HERE FOR CITYSCAPE
        from models.DRNSegDepth import DRNSegDepth
        model = DRNSegDepth(args.arch,
                            classes=19,
                            pretrained_model=None,
                            pretrained=False,
                            tasks=args.task_set)

        # CAN TEST THE MODEL HERE BY LOOPING THROUGH THE MODULES AND PASSING THE INPUT



    if args.pretrained and args.loading:
        print('args.pretrained', args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))

    # out_dir = 'output/{}_{:03d}'.format(args.arch, 0)

    print("including the following tasks:", taskonomy_tasks)

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            # print(p.size())
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print("Model has", get_n_params(model), "parameters")
    print("Encoder has", get_n_params(model.encoder), "parameters")

    for decoder in model.task_to_decoder.values():
        print("Decoder has", get_n_params(decoder), "parameters")

    model = torch.nn.DataParallel(model)


    if torch.cuda.is_available():
        model.cuda()

    # Data loading code
    info = get_info(args.dataset)
    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val", out_name=True)
    adv_val_loader = get_loader(args, "adv_val", out_name=True)

    # define loss function (criterion) and optimizer
    if args.optim == 'sgd':
        print("Using SGD")
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        print("Using Adam")
        optimizer = torch.optim.Adam(model.parameters())

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # Backup files before resuming/starting training
    backup_output_dir = args.backup_output_dir

    os.makedirs(backup_output_dir, exist_ok=True)

    if os.path.exists(backup_output_dir):
        import uuid
        unique_str = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_name = "train_" + args.arch + "_" + args.dataset + "_" + timestamp + "_" + unique_str\
                                   + "_trainset_{}_testset_{}_lambda_{}_seed_{}_lrs_{}_{}".format(args.class_to_train, args.class_to_test, args.mt_lambda, args.seed, args.step_size_schedule[1][0], args.step_size_schedule[2][0])
        if args.equally:
            experiment_name = experiment_name + "_equal"
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_name)
        print(experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder, ignore=include_patterns('*.py', '*.json'))


    experiment = Experiment(api_key="5cU4pCUJ2rWYAEAIZfAO01I3e",
                                 project_name="robustseg", workspace="vikramnitin9",
                                 auto_param_logging=False, auto_metric_logging=False,
                                 parse_args=False, display_summary=False, disabled=(not args.comet))

    experiment.set_name(experiment_name)
    experiment.log_parameters(vars(args))

    # Logging with TensorBoard
    log_dir = os.path.join(experiment_backup_folder, "runs")

    # os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    eval_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')

    fh = logging.FileHandler(experiment_backup_folder+'/log.txt')
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


    for epoch in range(start_epoch, args.epochs):

        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch

        torch.cuda.empty_cache()

        if args.adv_train:
            from learning.mtask_adv_train_loop import adv_train
            adv_train(train_loader, model, criteria, optimizer, epoch, writer, info, args.dataset,
                  eval_score=accuracy, args=args)
        else:
            train(train_loader, model, criteria, optimizer, epoch, writer, info, args.dataset,
                  comet=experiment, eval_score=accuracy, args=args)

        # evaluate on validation set
        total_loss = mtask_validate(val_loader, model, criteria, eval_writer, args=args,
                                    comet=experiment, eval_score=accuracy, info=info, epoch=epoch, print_freq=300)

        if args.adv_val and epoch % args.val_freq ==0:
            from learning.mtask_grad import mtask_forone_grad
            grad = mtask_forone_grad(val_loader, model, criteria, args.test_task_set, args)
            writer.add_scalar('Val/adv_Gradient', grad, epoch)

            from learning.mtask_grad import mtask_forone_advacc
            mtask_forone_advacc(val_loader, model, criteria, args.test_task_set, args, info, writer, epoch)

        is_best = total_loss < best_prec1  # fix the bug
        best_prec1 = min(total_loss, best_prec1)
        save_model_path = os.path.join(experiment_backup_folder, 'savecheckpoint')
        os.makedirs(save_model_path, exist_ok=True)
        checkpoint_path = os.path.join(save_model_path, 'checkpoint_latest.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path, save_model_path = save_model_path)
        if (epoch + 1) % 50 == 0:
            history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)

    from learning.mtask_grad import mtask_forone_advacc
    advacc_result = mtask_forone_advacc(val_loader, model, criteria, args.test_task_set, args, info, epoch,
                                        comet=experiment, norm='Linf', writer=eval_writer)

    print(advacc_result)

    writer.close()


def train(train_loader, model, criteria, optimizer, epoch, writer, info, dataset,
          comet=None, eval_score=None, print_freq=500, args=None):
    if args.debug:
        print_freq = 10
    # print('train 1')

    batch_time = AverageMeter()
    data_time = AverageMeter()

    avg_losses = {}
    for c_name, criterion_fun in criteria.items():
        avg_losses[c_name] = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    print('train len', len(train_loader))
    for i, (input, target, mask) in enumerate(train_loader):
        # measure data loading time
        if args.debug:
            print("input tr", input.size())
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            for keys, tar in target.items():
                target[keys] = tar.cuda()
            for keys, m in mask.items():
                mask[keys] = m.cuda()

        input_var = torch.autograd.Variable(input)

        # compute output
        loss_dict = {}

        output = model(input_var)

        # loss = criteria(output, target_var)
        sum_loss = None

        for c_name, criterion_fun in criteria.items():
            try:
                this_loss = criterion_fun(output[c_name].float(), target[c_name], mask[c_name])
            except:
                import pdb; pdb.set_trace()

            if args.equally:
                this_loss = this_loss * 1.0 / len(args.task_set)
            else:
                if c_name in args.test_task_set:
                    this_loss *= 1.0/len(args.test_task_set)
                else:
                    if (len(args.task_set) - len(args.test_task_set)) != 0:
                        this_loss *= args.mt_lambda/(len(args.task_set) - len(args.test_task_set))

            if sum_loss is None:
                sum_loss = this_loss
            else:
                sum_loss = sum_loss + this_loss

            loss_dict[c_name] = this_loss
            avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))


        loss = sum_loss  #TODO: remove clone here, why do they use clone?  Because of accumulating the gradient

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed timetrain
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:

            str = 'Epoch: [{0}][{1}/{2}]\t  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.\
                format( epoch, i, len(train_loader), batch_time=batch_time)
            for keys, loss_term in loss_dict.items():
                str += 'Loss : {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])

            logger.info(str)
            #TODO: a loop like below, which draw the prediction map and groundtruth map, in the tensorboard

            for keys, loss_term in loss_dict.items():
                writer.add_scalar('Train/Score {}'.format(keys), avg_losses[keys].val, epoch * len(train_loader) + i)
                if comet is not None: comet.log_metric('Train/Score {}'.format(keys), avg_losses[keys].val, step=epoch * len(train_loader) + i)

            # Show TensorBoard visualisations for image labels and the model predictions.
            # Show clean image
            writer.add_image('Train/Image_Clean', back_transform(input_var, info)[0])
            # if comet is not None: comet.log_image(back_transform(input_var, info)[0].cpu(), name='Train/Image_Clean', image_channels='first')
            # Show targets and predictions for each task
            for task_name, _ in criteria.items():
                # Only show for valid task names
                if task_name != "mask" and task_name!="rgb":
                    # Show tensorboard visualisations related to segmentation as we need to decode the labels to corresponding colors.
                    if task_name == "segmentsemantic":
                        class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                        decoded_target = decode_segmap(
                            target['segmentsemantic'][0][0].cpu().data.numpy() if torch.cuda.is_available() else
                            target['segmentsemantic'][0][0].data.numpy(),
                            args.dataset)
                        image_label = np.moveaxis(decoded_target, 2, 0)
                        decoded_class_prediction = decode_segmap(
                            class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else class_prediction[
                                0].data.numpy(), args.dataset)
                        task_prediction = np.moveaxis(decoded_class_prediction, 2, 0)
                    elif task_name == 'autoencoder':
                        transformed_image_label = back_transform(target[task_name], info)
                        transformed_task_prediction = back_transform(output[task_name], info)
                        # image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else target[task_name][0].data.numpy()
                        image_label = transformed_image_label[0].cpu().data.numpy() if torch.cuda.is_available() else transformed_image_label[0].data.numpy()
                        task_prediction = transformed_task_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else transformed_task_prediction[0].data.numpy()
                    elif task_name == 'depth_zbuffer':
                        image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else target[task_name][0].data.numpy()
                        task_prediction = output[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else output[task_name][0].data.numpy()
                    else:
                        image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else target[task_name][0].data.numpy()
                        task_prediction = output[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else output[task_name][0].data.numpy()

                    if image_label.shape[0] != 2 and len(image_label.shape) == 3:
                        group_image_label_and_prediction = np.stack((image_label, task_prediction))
                        writer.add_images('Train/Image_label_and_prediction/{}'.format(task_name), group_image_label_and_prediction)

            if args.debug and i==print_freq:
                break


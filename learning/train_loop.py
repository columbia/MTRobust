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

from dataloaders.utils import decode_segmap

from torch.utils.tensorboard import SummaryWriter
import torchvision

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained and args.loading:
        print('args.pretrained', args.pretrained)
        single_model.load_state_dict(torch.load(args.pretrained))

    out_dir = 'output/{}_{:03d}_{}'.format(args.arch, 0, args.phase)

    model = torch.nn.DataParallel(single_model)

    if args.select_class:
        weight_add = torch.ones((args.classes), dtype=torch.float32).cuda() \
            if torch.cuda.is_available() else torch.ones((args.classes) , dtype=torch.float32)
        for each in args.train_category:
            # weight_add[each] = args.weight_mul / (len(args.train_category) * 1.0)
            weight_add[each] = args.weight_mul
        criterion = nn.NLLLoss(weight=weight_add, ignore_index=255)
    else:
        criterion = nn.NLLLoss(ignore_index=255)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # Data loading code
    info = get_info(args.dataset)
    train_loader = get_loader(args,"train")
    val_loader = get_loader(args,"val", out_name=True)
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
        experiment_backup_folder = "train_" + args.arch + "_" + args.dataset + "_" + timestamp
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_backup_folder)
        print(experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder, ignore=include_patterns('*.py', '*.json'))


    # Logging with TensorBoard
    log_dir = os.path.join(experiment_backup_folder, "runs")

    # os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    val_writer = SummaryWriter(log_dir=log_dir+'/validate_runs/')

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

    if args.evaluate:

        validate(val_loader, model, criterion, args=args ,log_dir=experiment_backup_folder, eval_score=accuracy,
                 info=info)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer, info, args.dataset,
              eval_score=accuracy, args=args)

        # # evaluate on validation set
        total_loss = validate(val_loader, model, criterion, args=args, log_dir=experiment_backup_folder, writer=val_writer,
                              eval_score=accuracy, info=info, epoch=epoch)

        #
        # from learning.validate import validate_adv
        #
        # if args.adv_val:
        #     mAP = validate_adv(adv_val_loader, model, args.classes, save_vis=True,
        #                    has_gt=True, output_dir=out_dir, downsize_scale=args.downsize_scale, log_dir=experiment_backup_folder,
        #                    args=args, info=info, epoch=epoch)
        #     writer.add_scalar('Validate/adv_acc', mAP, epoch)
        #
        #

        is_best = total_loss > best_prec1
        best_prec1 = max(total_loss, best_prec1)
        save_model_path = os.path.join(experiment_backup_folder, 'savecheckpoint')
        os.makedirs(save_model_path, exist_ok=True)
        checkpoint_path = os.path.join(save_model_path, 'checkpoint_latest.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path, save_model_path = save_model_path)
        if (epoch + 1) % 5 == 0:
            # history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            # history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            history_path = os.path.join(save_model_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, info, dataset,
          eval_score=None, print_freq=10, args=None):
    if args.debug:
        print_freq = 10

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.select_class:
        print('class under training', args.train_category)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if args.debug:
            print("input tr", input.size())
        data_time.update(time.time() - end)

        # temp = torch.sum(target == torch.ones_like(target).long() * 11).item()
        # if temp>0:
        #     print('before processing', temp)

        # mm = np.histogram(target.numpy(),bins=[mmm for mmm in range(19)])
        # print('tar hist', mm)

        if args.select_class:
            for tt, each in enumerate(args.train_category):
                if tt == 0:
                    # print('target size', target.size())
                    mask_as_none = 1 - (torch.ones_like(target).long() * each == target).long()
                else:
                    mask_as_none = mask_as_none.long() * (1 - (torch.ones_like(target).long() * each == target).long())
            target = target * (1-mask_as_none.long()) + args.others_id * torch.ones_like(target).long() * mask_as_none.long()

        # temp = torch.sum(target == torch.ones_like(target).long() * 1).item()
        # if temp>0:
        #     print('found one', temp)

        # print(torch.sum((target == torch.ones_like(target).long() * 7) + (target == torch.ones_like(target).long() * 11)))
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # print('loss', loss, loss.data, loss.data.item(), input.size(0))
        losses.update(loss.data.item(), input.size(0))
        if eval_score is not None:
            if target_var.size(0)>0:
                if args.calculate_specified_only:
                    for tt, each in enumerate(args.train_category):
                        if tt == 0:
                            # print('target size', target.size())
                            mask_as_none = 1 - (torch.ones_like(target).long() * each == target).long()
                        else:
                            mask_as_none = mask_as_none.long() * (
                                        1 - (torch.ones_like(target).long() * each == target).long())

                    target_temp = target_var * (1 - mask_as_none.long()) + 255 * torch.ones_like(target).long() * mask_as_none.long()
                    scores.update(eval_score(output, target_temp), input.size(0))
                else:
                    scores.update(eval_score(output, target_var), input.size(0))
            else:
                print("0 size!")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed timetrain
        batch_time.update(time.time() - end)
        end = time.time()


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
            # print("____info is ",info)
            writer.add_image('Image/image ', back_transform(input_var, info)[0])
            writer.add_image('Image/image target ', decoded_target)
            writer.add_image('Image/image prediction ', decoded_class_prediction)
            writer.add_scalar('Train/Score', scores.val, epoch * len(train_loader) + i)
            writer.add_scalar('Train/Loss', losses.val, epoch * len(train_loader) + i)

            if args.debug and i==print_freq:
                break




# input.shape torch.Size([4, 3, 512, 512]) tensor(-1.4918) tensor(3.8802)
# input_var.shape torch.Size([4, 3, 512, 512]) tensor(-1.4918) tensor(3.8802)
# target.shape torch.Size([4, 512, 512]) tensor(0) tensor(255)
# target_var torch.Size([4, 512, 512]) tensor(0) tensor(255)

import models.drn as drn
from models.DRNSeg import DRNSeg
from models.FCN32s import FCN32s
import data_transforms as transforms
import json
import math
import os
from os.path import exists, join, split
import threading

import time

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

from torch.utils.tensorboard import SummaryWriter

from learning.dataloader import SegList, SegListMS
from learning.utils_learn import *
from learning.model_config import *

import data_transforms as transforms
from dataloaders.utils import decode_segmap

import logging
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#TODO: validate and adv_validate are not very consistent. The error and loss calculation and also the arguments and their order are different. eg- args is not passed in validate().
#TODO: 26Sep has_gt might not be everywhere where validate() is
def validate(val_loader, model, criterion, args=None, log_dir=None, eval_score=None, print_freq=200, info=None,
             writer=None, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    print("___Entering Validation validate()___")

    # switch to evaluate mode
    model.eval()

    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))

    end = time.time()
    for i, (input, target, name) in enumerate(val_loader):

        if args.select_class:
            for tt, each in enumerate(args.train_category):
                if tt == 0:
                    # print('target size', target.size())
                    mask_as_none = 1 - (torch.ones_like(target).long() * each == target).long()
                else:
                    mask_as_none = mask_as_none.long() * (1 - (torch.ones_like(target).long() * each == target).long())
            target = target * (1-mask_as_none.long()) + args.others_id * torch.ones_like(target).long() * mask_as_none.long()


        if args.debug:
            print("nat validate size", input.size())
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input) # TODO: volatile keyword is deprecated, so use with torch.no_grad():; see if necessary
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        if eval_score is not None:

            if args.calculate_specified_only:
                for tt, each in enumerate(args.train_category):
                    if tt == 0:
                        # print('target size', target.size())
                        mask_as_none = 1 - (torch.ones_like(target).long() * each == target).long()
                    else:
                        mask_as_none = mask_as_none.long() * (
                                1 - (torch.ones_like(target).long() * each == target).long())

                target_temp = target_var * (1 - mask_as_none.long()) + 255 * torch.ones_like(
                    target).long() * mask_as_none.long()
                score.update(eval_score(output, target_temp), input.size(0))
            else:

                score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        class_prediction = torch.argmax(output, dim=1)
        decoded_target = decode_segmap(target_var[0].cpu().data.numpy() if torch.cuda.is_available() else target_var[0].data.numpy(), args.dataset)
        decoded_target = np.moveaxis(decoded_target, 2, 0)
        decoded_class_prediction = decode_segmap(class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else class_prediction[0].data.numpy(), args.dataset)
        decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)

        target = target.cpu().data.numpy() if torch.cuda.is_available() else target.data.numpy()
        class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
        hist += fast_hist(class_prediction.flatten(), target.flatten(), num_classes)

        if args.debug:
            print_freq = 10
        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

            # Visualise the images
            if writer is not None:
                print("_____Beginning to write images to tfb now")
                writer.add_image('Val/image ', back_transform(input_var, info)[0])
                writer.add_image('Val/image target ', decoded_target)
                writer.add_image('Val/image prediction ', decoded_class_prediction)

            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))

            if args.debug and i==print_freq:
                break

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    ious = per_class_iu(hist) * 100
    logger.info(' '.join('{:.03f}'.format(i) for i in ious))

    if writer is not None:
        writer.add_scalar('Val Clean/ Seg mIoU', round(np.nanmean(ious), 2), epoch)
        writer.add_scalar('Val Clean/ Seg Accuracy', score.avg, epoch)

    return round(np.nanmean(ious), 2)




##TODO: designed solely for test
from learning.attack import PGD_attack
def validate_adv_test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None, info=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))

    criterion = nn.NLLLoss(ignore_index=255)

    for iter, (image, label, name) in enumerate(eval_data_loader):
        # the input here are already normalized ,
        if args.debug:
            print('adv val', image.size())
        data_time.update(time.time() - end)
        # print('input', torch.max(image), torch.min(image))
        # old_in = back_transform(image, info)
        # print('input back trans', torch.max(old_in), torch.min(old_in))

        # crop image to see if works
        im_height = image.size(2)
        im_width = image.size(3)
        downsize_scale = downsize_scale  # change with a flag
        if (downsize_scale != 1):
            print("down scale")
            im_height_downsized = (int)(im_height / downsize_scale)
            im_width_downsized = (int)(im_width / downsize_scale)

            delta_height = (im_height - im_height_downsized) // 2
            delta_width = (im_width - im_width_downsized) // 2
            # print("Image sizes before and after downsize",im_height, im_width, im_height_downsized, im_width_downsized)
            image = image[:, :, delta_height:im_height_downsized + delta_height,
                    delta_width:im_width_downsized + delta_width]
            label = label[:, delta_height:im_height_downsized + delta_height,
                    delta_width:im_width_downsized + delta_width]

        adv_img = PGD_attack(image, label, model, criterion, args.epsilon, args.steps, args.dataset,
                             args.step_size, info, using_noise=True)

        # TODO: Move variables to CUDA - see adv_train
        image_var = Variable(adv_img.data, requires_grad=False)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        # if save_vis:
        #     # print('save', output_dir)
        #     save_output_images(pred, name, output_dir + "_nat")
        #     save_colorful_images(
        #         pred, name, output_dir + 'adv__color',
        #         TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            pred = pred.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
        # break

    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def validate_adv(eval_data_loader, model, num_classes, log_dir=None,
         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None, info=None, eval_score=None,
                 writer=None, epoch=0):
    """
    Function for validation with adversarial images.
    :param eval_data_loader:
    :param model:
    :param num_classes:
    :param log_folder: Directory path to save Tensorflow Board.
    :param output_dir:
    :param has_gt:
    :param save_vis:
    :param downsize_scale:
    :param args:
    :param info:
    :return:
    """

    print("___Entering Adversarial Validation validate_adv()___")

    score = AverageMeter()

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))

    criterion = nn.NLLLoss(ignore_index=255)


    for iter, (image, label, name) in enumerate(eval_data_loader):

        # print("_____iteration in validate is: ", iter)

        # print('validate_adv img size', image.size())
        # the input here are already normalized ,
        data_time.update(time.time() - end)

        if args.select_class:
            for tt, each in enumerate(args.train_category):
                if tt == 0:
                    # print('target size', target.size())
                    mask_as_none = 1 - (torch.ones_like(label).long() * each == label).long()
                else:
                    mask_as_none = mask_as_none.long() * (1 - (torch.ones_like(label).long() * each == label).long())
            label = label * (1-mask_as_none.long()) + args.others_id * torch.ones_like(label).long() * mask_as_none.long()


        #TODO: Categorise and define variables properly
        adv_img = PGD_attack(image, label, model, criterion, args.epsilon, args.steps, args.dataset,
                             args.step_size, info, using_noise=True)
        # print("_____adv_img calculated")

        clean_input = image
        input = adv_img.data
        # TODO: Move variables to CUDA - see adv_train
        if torch.cuda.is_available(): #only input is necessary to be put on cuda
            input = input.cuda()
            # label = label.cuda()
            # clean_input = clean_input.cuda()

        # print("_____putting in model")

        input_var = Variable(input, requires_grad=False) #TODO: volatile is removed, use with torch.no_grad() if necessary
        final = model(input_var)[0]
        # print("_____out of model")
        _, pred = torch.max(final, 1)
        batch_time.update(time.time() - end)

        decoded_target = decode_segmap(label[0].cpu().data.numpy() if torch.cuda.is_available() else label[0].data.numpy(), args.dataset)
        decoded_target = np.moveaxis(decoded_target, 2, 0)
        decoded_class_prediction = decode_segmap(pred[0].cpu().data.numpy() if torch.cuda.is_available() else pred[0].data.numpy(), args.dataset)
        decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)


        if eval_score is not None:

            if args.calculate_specified_only:
                for tt, each in enumerate(args.train_category):
                    if tt == 0:
                        # print('target size', target.size())
                        mask_as_none = 1 - (torch.ones_like(label).long() * each == label).long()
                    else:
                        mask_as_none = mask_as_none.long() * (
                                1 - (torch.ones_like(label).long() * each == label).long())

                target_temp = label * (1 - mask_as_none.long()) + 255 * torch.ones_like(
                    label).long() * mask_as_none.long()
                score.update(eval_score(final, target_temp), input.size(0))
            else:

                score.update(eval_score(final, label), input.size(0))


        # if save_vis:
        #     # print('save', output_dir)
        #     save_output_images(pred, name, output_dir + "_nat")
        #     save_colorful_images(
        #         pred, name, output_dir + 'adv__color',
        #         TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)

        label = label.numpy()
        pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
        hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

        end = time.time()

        freq_print = args.print_freq
        if args.debug:
            freq_print = 10
        if iter % freq_print == 0:
            logger.info('Eval: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(iter, len(eval_data_loader), batch_time=batch_time,
                                data_time=data_time))


            # Visualise the images
            if writer is not None:
                print("_____Beginning to write now")
                writer.add_image('Val_Adv/adv image ', back_transform(input_var, info)[0])
                writer.add_image('Val_Adv/clean image ', back_transform(image, info)[0])
                writer.add_image('Val_Adv/image target ', decoded_target)
                writer.add_image('Val_Adv/image prediction ', decoded_class_prediction)

            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))

            logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

            if args.debug and iter == freq_print:
                #breaks after print_freq number of batches.
                break

    logger.info(' *****\n***OverAll***\n Score {top1.avg:.3f}'.format(top1=score))

    ious = per_class_iu(hist) * 100
    logger.info(' '.join('{:.03f}'.format(i) for i in ious))

    if writer is not None:
        writer.add_scalar('Val Adv/ Seg mIoU', round(np.nanmean(ious), 2), epoch)
        writer.add_scalar('Val Adv/ Seg Accuracy', score.avg, epoch)
    return round(np.nanmean(ious), 2)

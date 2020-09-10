#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This code is the demo for the robustness under multitask attack on segmentation model.

import argparse
import json
import logging
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
from torchvision import datasets, transforms
from torch.autograd import Variable


import models.drn as drn
from models.DRNSeg import DRNSeg
from models.FCN32s import FCN32s
import data_transforms as transforms


try:
    from modules import batchnormsync
except ImportError:
    pass

from learning.train_loop import *
from learning.validate import *
from learning.test import *

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_torch_std(info):
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    return tensor_std

def PGD_drnseg_masked_attack_city(x, y, attack_mask, net, criterion, epsilon, steps, step_size, info):
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag = True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.

    ones_like_x_adv = torch.ones_like(x_adv)

    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()
        tensor_std = tensor_std.cuda()

    pert_epsilon = ones_like_x_adv * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    # TODO: print and check the bound

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        # Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)  # dict{rep:float32,segmentasemantic:float32, depth_zbuffer:float32, reconstruct:float32}

        ignore_value = 255
        attack_mask = attack_mask.long()

        y = y * attack_mask + ignore_value * (1 - attack_mask)  # y is {auto:float,segsem:int64,deoth:float}
        grad_total_loss = criterion(h_adv[0], y)

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        x_adv = Variable(x_adv.data, requires_grad=True)  # TODO: optimize, remove this variable init each

    return x_adv


def test_drnseg_masked_attack(eval_data_loader, model):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    # hist = np.zeros((num_classes, num_classes))
    # exit(0)

    if torch.cuda.is_available():
        GPU_flag = True
    else:
        GPU_flag = False

    # Number of points to be selected for masking - analogous to number of output dimensions. Only these many pixels will be considered to calculate the loss.
    select_num_list = [1, 2, 5, 10, 50, 100, 200] + [i * 500 for i in range(1, 15)]

    result_list = []
    for select_num in select_num_list:
        print("********")
        print("selecting {} of output".format(select_num))
        import random
        acc_sample_avg_sum = 0
        if select_num < 400:
            MCtimes = 3
        else:
            MCtimes = 1
        MCtimes = 1
        # Monte Carlo Sampling - MCTimes is the number of times that we sample
        for inner_i in range(MCtimes):
            acc_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, name) in enumerate(eval_data_loader):
                # break if 50 images (batches) done to speed up the demo
                if cnt > 50:
                    break

                data_time.update(time.time() - end)

                if torch.cuda.is_available():
                    image_var = Variable(image.cuda(), requires_grad=True)
                else:
                    image_var = Variable(image, requires_grad=True)

                    # Generate mask for attack
                # For this image, sample select_num number of pixels
                temp = [i for i in range(image_var.size(2) * image_var.size(3))]
                selected = random.sample(temp, select_num)
                attack_mask = np.zeros((image_var.size(2) * image_var.size(3)), dtype=np.uint8)
                for iii in range(select_num):
                    attack_mask[selected[iii]] = 1

                attack_mask = attack_mask.reshape(1, 1, image_var.size(2), image_var.size(3))
                attack_mask = torch.from_numpy(attack_mask)

                if GPU_flag:
                    attack_mask = attack_mask.cuda()
                    label = label.cuda()

                attack_mask = Variable(attack_mask)

                # TODO - get the things reqd for its arguments such as criteria and tasks
                info = {"mean": [0.29010095242892997,0.32808144844279574,0.28696394422942517],
                        "std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]}
                criteria = cross_entropy2d
                adv_image_var = PGD_drnseg_masked_attack_city(image_var, label, attack_mask, model, criteria,
                                                              epsilon=8, steps=5,
                                                              step_size=2, info=info)

                final = model(adv_image_var)[0]
                _, pred = torch.max(final, 1)

                def accuracy_masked_attack(preds, label, mask):
                    valid_label = (label >= 0) * (label <= 18)
                    valid = valid_label * mask.bool()
                    acc_sum = (valid * (preds == label)).sum()
                    valid_sum = valid.sum()
                    acc = float(acc_sum) / (valid_sum + 1e-10)
                    return acc

                acc = accuracy_masked_attack(pred, label, attack_mask)
                acc_sum += acc
                cnt += 1
                batch_time.update(time.time() - end)
                end = time.time()

            acc_batch_avg = acc_sum / cnt  # Represents the gradient average for batch. cnt is the number of samples in a batch.
            acc_sample_avg_sum += acc_batch_avg  # For each sampling this is the sum of avg gradients in that sample.

        acc_sample_avg_sum /= MCtimes

        result_list.append(acc_sample_avg_sum)

        print(select_num, 'middle result', result_list)

    print('Final', result_list)

def test_grad_diffoutdim(eval_data_loader, model):
    """
    Evaluates the effect of increasing output dimension on the norm of the gradient.
    Monte Carlo sampling will be used and the result would be averaged.
    First choose the number of pixels to calculate the loss for (output dimension) --> select_num.
    For each select_num, we do the following MC_times(as Monte Carlo sampling):
        Calculate the loss for select_num pixels chosen, backpropagate and get the input gradient.
    Average all these.


    :param eval_data_loader:
    :param model:
    :param num_classes:
    :param output_dir:
    :param has_gt:
    :param save_vis:
    :param downsize_scale:
    :param args:
    :return:
    """
    model.eval()

    if torch.cuda.is_available():
        GPU_flag = True
    else:
        GPU_flag = False

    # Number of points to be selected for masking - analogous to number of output dimensions. Only these many pixels will be considered to calculate the loss.
    select_num_list = [1] + [i * 4 for i in range(1, 100)] + [400 + i*200 for i in range(100)]

    result_list = []
    for select_num in select_num_list:
        print("********")
        print("selecting {} of output".format(select_num))
        import random
        grad_sample_avg_sum = 0

        MCtimes = 20
        MCtimes = 1  # for demo speed up
        # Monte Carlo Sampling - MCTimes is the number of times that we sample
        for inner_i in range(MCtimes):
            grad_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, name) in enumerate(eval_data_loader):

                # break if 50 images (batches) done
                if cnt > 200:
                    break

                if torch.cuda.is_available():
                    image_var = Variable(image.cuda(), requires_grad=True)
                else:
                    image_var = Variable(image, requires_grad=True)
                final = model(image_var)[0]
                _, pred = torch.max(final, 1)

                # for this image, sample select_num number of pixels
                temp = [i for i in range(image_var.size(2) * image_var.size(3))]
                selected = random.sample(temp, select_num)

                # Build mask for image -
                mask = np.zeros((image_var.size(2) * image_var.size(3)), dtype=np.uint8)
                for iii in range(select_num):
                    mask[selected[iii]] = 1
                mask = mask.reshape(1, 1, image_var.size(2), image_var.size(3))
                mask = torch.from_numpy(mask)
                mask = mask.float()
                mask_target = mask.long()

                label = label.long()
                if GPU_flag:
                    mask = mask.cuda()
                    mask_target = mask_target.cuda()
                    label = label.cuda()

                target, mask = Variable(label), Variable(mask)
                loss = cross_entropy2d(final * mask, target * mask_target, size_average=False)
                loss.backward()

                data_grad = image_var.grad
                np_data_grad = data_grad.cpu().numpy()
                L2_grad_norm = np.linalg.norm(np_data_grad) / select_num # the 1/M \sum_M \partial{Loss_i}/\partial{input}
                grad_sum += L2_grad_norm
                # increment the batch # counter
                cnt += 1

            grad_avg = grad_sum / cnt # Represents the gradient average for batch. cnt is the number of samples in a batch.
            grad_sample_avg_sum += grad_avg # For each sampling this is the sum of avg gradients in that sample.

        grad_sample_avg_sum /= MCtimes

        result_list.append(grad_sample_avg_sum)

        print(select_num, 'graident L2 Norm List:', result_list)
        np.save('grad_mtl.npy', result_list)

    print('Final', result_list)
    np.save('grad_mtl.npy', result_list)

def test_seg(data_dir, modeldir, test_acc_output_dim, get_grad):
    test_batch_size =2

    single_model = DRNSeg("drn_d_22", 19, pretrained_model=None,
                          pretrained=False)
    single_model.load_state_dict(torch.load(modeldir))
    model = torch.nn.DataParallel(single_model)
    if torch.cuda.is_available():
        model.cuda()

    normalize = transforms.Normalize(mean=[0.29010095242892997,0.32808144844279574,0.28696394422942517],
                                     std=[0.1829540508368939, 0.18656561047509476, 0.18447508988480435])

    test_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), list_dir=None, out_name=True),
        batch_size=test_batch_size, shuffle=False, num_workers=8,
        pin_memory=True, drop_last=True
    )
    if test_acc_output_dim:
        test_drnseg_masked_attack(test_loader, model)
    elif get_grad:
        _ = test_grad_diffoutdim(test_loader, model)


from learning.adv_train_loop import train_seg_adv
def main():
    datadir = '/local/rcs/ECCV/Cityscape/cityscape_dataset'
    modeldir = '/local/rcs/ECCV/Cityscape/model_zoo/DRN/drn_d_22_cityscapes.pth'
    test_seg(datadir, modeldir, test_acc_output_dim=False, get_grad=True)


if __name__ == '__main__':
    main()

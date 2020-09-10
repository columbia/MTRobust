import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading
import shutil
from fnmatch import filter

from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# added for adversarial experiment
import torch.nn.functional as F
from distutils.version import LooseVersion

import numpy as np

def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image

def back_transform(image, info):
    # image = image2.copy()

    image[:, 0, :, :] = image[:, 0, :, :] * info["std"][0] + info["mean"][0]
    image[:, 1, :, :] = image[:, 1, :, :] * info["std"][1] + info["mean"][1]
    image[:, 2, :, :] = image[:, 2, :, :] * info["std"][2] + info["mean"][2]
    return image

def forward_transform(image, info):
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out



def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (args.lr_change ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    elif args.lr_mode == 'schedule':
        print('args.args.step_size_schedule',args.step_size_schedule)
        assert len(args.step_size_schedule) == 3
        lr = args.step_size_schedule[0][1]
        if epoch >= args.step_size_schedule[1][0] and epoch < args.step_size_schedule[2][0]:
            lr = args.step_size_schedule[1][1]
        elif epoch >= args.step_size_schedule[2][0]:
            lr = args.step_size_schedule[2][1]

    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_model_path = None):
    try:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(save_model_path, 'model_best.pth.tar'))
    except:
        for _ in range(30): print("--------------WARNING!!! FAILED TO SAVE. DISK POSSIBLY OUT OF SPACE--------------")
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    if correct.size(0) == 0:
        pass
        # print('c1', correct.size())
    correct = correct[target != 255]
    correct = correct.view(-1)
    if correct.size(0) == 0:
        # print('c2', correct.size(), correct)
        cor_num = correct.float().sum(0)
        score = cor_num.mul(100.0 / 1)
    else:
        cor_num = correct.float().sum(0)
        # print('correc size', correct.size(0))
        score = cor_num.mul(100.0 / correct.size(0))
    # print('cor num', cor_num, correct.size(0),correct.size())
    # return score.data[0]
    return score.data.item()


def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index=255):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum', ignore_index=ignore_index)
    if size_average:
        loss /= mask.data.sum()
    return loss
# added for adversarial experiment ends

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """

    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                   for name in filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not os.path.isdir(join(path, name)))
        return ignore

    return _ignore_patterns




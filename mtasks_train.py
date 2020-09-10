import argparse
import os
import shutil
import time
import platform
import random

from comet_ml import Experiment

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets


import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
import scipy.stats

# from ptflops import get_model_complexity_info

import models.taskonomy_models as models
#, "depth_zbuffer", "edge_texture", "keypoints2d"
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')

    parser.add_argument('--dataset',        type=str,   required=True)
    parser.add_argument('--model',          type=str,   required=True)
    parser.add_argument('--loading',        action='store_true')
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--debug',          action='store_true')
    parser.add_argument('--equally',          action='store_true')
    parser.add_argument('--customize_class',action='store_true')
    parser.add_argument('--comet',          action='store_true')
    parser.add_argument('--class_to_train', type=str)
    parser.add_argument('--class_to_test',  type=str)
    parser.add_argument('--seed',           type=int,       default=42,     help='seed')
    parser.add_argument('--mt_lambda',      type=float,     default=0,      help='mt_lambda')
    parser.add_argument('--step_size_schedule',type=str,    default='[[0, 0.01], [140, 0.001], [200, 0.0001]]', help='lr schedule')
    parser.add_argument('--optim',          type=str,       default='sgd',  help='sgd/adam')


    cudnn.benchmark = False
    args = parser.parse_args()
    import socket, json
    config_file_path = "config/{}_{}_config.json".format(args.model, args.dataset)
    with open (config_file_path) as config_file:
        config = json.load(config_file)
    if socket.gethostname() == "deep":
        args.data_dir = config['data-dir_deep']
        args.pretrained = config['pretrained_deep']
        args.backup_output_dir = config['backup_output_dir_deep']
    elif socket.gethostname() == "amogh":
        args.data_dir = config['data-dir_amogh']
        args.pretrained = config['pretrained_amogh']
        args.backup_output_dir = config['backup_output_dir_amogh']
    elif socket.gethostname() == 'cv04':
        args.data_dir = config['data-dir_cv04']
        args.pretrained = ""
        args.backup_output_dir = config['backup_output_dir_cv04']
    elif socket.gethostname() == 'hulk':
        args.data_dir = config['data-dir_hulk']
        args.pretrained = ""
        args.backup_output_dir = config['backup_output_dir_hulk']
    else:
        args.data_dir = config['data-dir']
        args.pretrained = config['pretrained']
        args.backup_output_dir = config['backup_output_dir']

    args.step = config['step']
    args.arch = config['arch']
    args.batch_size = config['batch-size']
    args.test_batch_size = config['test-batch-size']
    args.epochs = config['epochs']
    # args.mt_lambda = config['mt_lambda']


    args.lr_change = config['lr_change']
    args.lr = config['lr']
    args.lr_mode = config['lr-mode']
    args.momentum = config['momentum']
    args.weight_decay = config['weight-decay']

    import ast
    args.step_size_schedule = ast.literal_eval(args.step_size_schedule)

    args.workers = config['workers']

    args.print_freq = config['print_freq']

    args.classes = config['classes']



    # ADDED FOR CITYSCAPES
    args.random_scale = config['random-scale']
    args.random_rotate = config['random-rotate']
    args.crop_size = config['crop-size']
    args.list_dir = config['list-dir']

    if args.customize_class:  # TODO: Notice here the sequence of each task is hard coded, during the testing, this sequence must be fully followed.
        # TODO: because the nn.ModuleList does not have the key for each decoder, thus if decoder is swithched sequence during loading, error will occur,
        # Even if no error is raised, the decoder is loaded with wrong weights thus results would be wrong.

        t_list = []
        if 's' in args.class_to_train:
            t_list.append("segmentsemantic")
        if 'd' in args.class_to_train:
            t_list.append("depth_zbuffer")
        if 'e' in args.class_to_train:
            t_list.append("edge_texture")
        if 'k' in args.class_to_train:
            t_list.append("keypoints2d")
        if 'n' in args.class_to_train:
            t_list.append("normal")
        if 'r' in args.class_to_train:
            t_list.append("reshading")

        if 'K' in args.class_to_train:
            t_list.append("keypoints3d")
        if 'D' in args.class_to_train:
            t_list.append("depth_euclidean")
        if 'A' in args.class_to_train:
            t_list.append("autoencoder")
        if 'E' in args.class_to_train:
            t_list.append("edge_occlusion")
        if 'p' in args.class_to_train:
            t_list.append("principal_curvature")
        if 'u' in args.class_to_train:
            t_list.append("segment_unsup2d")
        if 'U' in args.class_to_train:
            t_list.append("segment_unsup25d")


        test_t_list = []
        if 's' in args.class_to_test:
            assert 's' in args.class_to_train
            test_t_list.append("segmentsemantic")
        if 'd' in args.class_to_test:
            test_t_list.append("depth_zbuffer")
        if 'e' in args.class_to_test:
            test_t_list.append("edge_texture")
        if 'k' in args.class_to_test:
            test_t_list.append("keypoints2d")
        if 'n' in args.class_to_test:
            test_t_list.append("normal")
        if 'r' in args.class_to_test:
            test_t_list.append("reshading")

        if 'K' in args.class_to_test:
            test_t_list.append("keypoints3d")
        if 'D' in args.class_to_test:
            test_t_list.append("depth_euclidean")
        if 'A' in args.class_to_test:
            test_t_list.append("autoencoder")
        if 'E' in args.class_to_test:
            test_t_list.append("edge_occlusion")
        if 'p' in args.class_to_test:
            test_t_list.append("principal_curvature")
        if 'u' in args.class_to_test:
            test_t_list.append("segment_unsup2d")
        if 'U' in args.class_to_test:
            test_t_list.append("segment_unsup25d")

        args.task_set = t_list
        args.test_task_set = test_t_list

    else:
        args.task_set = config['task_set']
        args.test_task_set = config['test_task_set']

    args.adv_val = config['adv_val']
    args.val_freq = config['val_freq']
    args.adv_train = False

    args.epsilon = config['epsilon']
    args.step_size = config['step_size']
    args.steps = config['steps']


    return args


# , "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"
def main():
    args = parse_args()
    # args.arch = 'res18'
    print('starting on', platform.node())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:', os.environ['CUDA_VISIBLE_DEVICES'])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        main_stream = torch.cuda.Stream()

        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

    from learning.mtask_train_loop import train_mtasks
    train_mtasks(args)

if __name__ == '__main__':
    main()

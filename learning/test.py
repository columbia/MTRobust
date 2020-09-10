### THis file is moved to eval.py, not use this anymore

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
import logging
from learning.dataloader import SegList, SegListMS, get_info,get_loader
from learning.utils_learn import *
from learning.attack import PGD_masked_attack_mtask_city, PGD_drnseg_masked_attack_city

import data_transforms as transforms



FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    # model specific
    model_arch = args.arch
    task_set_present = hasattr(args, 'task_set')
    # if (model_arch.startswith('drn')):
    #     if task_set_present:
    #         from models.DRNSegDepth import DRNSegDepth
    #         print("LENGTH OF TASK SET IN CONFIG>1 => LOADING DRNSEGDEPTH model for multitask, to load DRNSEG, remove the task_set from config args.")
    #         single_model = DRNSegDepth(args.arch,
    #                             classes=19,
    #                             pretrained_model=None,
    #                             pretrained=False,
    #                             tasks=args.task_set)
    #     else:
    #         single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
    #                           pretrained=False)
    # elif (model_arch.startswith('fcn32')):
    #     # define the architecture for FCN.
    #     single_model = FCN32s(args.classes)
    # else:
    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)  # Replace with some other model
    print("Architecture unidentifiable, please choose between : fcn32s, dnn_")

    if args.pretrained:
        print('args.pretrained', args.pretrained)
        single_model.load_state_dict(torch.load(args.pretrained))

    model = torch.nn.DataParallel(single_model)
    if torch.cuda.is_available():
        model.cuda()

    data_dir = args.data_dir

    # info = json.load(open(join(data_dir, 'info.json'), 'r'))
    # normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    # scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    # if args.ms:
    #     dataset = SegListMS(data_dir, phase, transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), scales, list_dir=args.list_dir)
    # else:
    #
    #     dataset = SegList(data_dir, phase, transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), list_dir=args.list_dir, out_name=True)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #     pin_memory=False
    # )

    test_loader = get_loader(args, phase,out_name=True)
    info = get_info(args.dataset)

    cudnn.benchmark = True

    # Backup files before resuming/starting training
    backup_output_dir = args.backup_output_dir
    os.makedirs(backup_output_dir, exist_ok=True)

    if os.path.exists(backup_output_dir):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        experiment_backup_folder = "test_" + args.arch + "_" + args.dataset + "_" + timestamp
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_backup_folder)
        os.makedirs(experiment_backup_folder)
        print(experiment_backup_folder)


    fh = logging.FileHandler(experiment_backup_folder + '/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Make sure the name of the dataset and model are included in the output file.
    out_dir = 'output/{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.adv_test:
        from learning.validate import validate_adv_test
        mAP = validate_adv_test(test_loader, model, args.classes, save_vis=True,
                       has_gt=True, output_dir=out_dir, downsize_scale=args.downsize_scale,
                       args=args, info=info)
    elif args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        if args.test_acc_output_dim:
            test_drnseg_masked_attack(test_loader, model, args.classes, save_vis=True,
                                 has_gt=phase != 'test' or args.with_gt, output_dir=out_dir,
                                 downsize_scale=args.downsize_scale,
                                 args=args)
            # test_masked_accuracy_outdim(test_loader, model, args.classes, save_vis=True,
            #                      has_gt=phase != 'test' or args.with_gt, output_dir=out_dir,
            #                      downsize_scale=args.downsize_scale,
            #                      args=args)


        else:
            mAP = test_grad_diffoutdim(test_loader, model, args.classes, save_vis=True,
                                       has_gt=phase != 'test' or args.with_gt, output_dir=out_dir, downsize_scale=args.downsize_scale,
                                       args=args)
    logger.info('mAP: %f', mAP)


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_grad_diffoutdim(eval_data_loader, model, num_classes,
                         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None):
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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    # exit(0)

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
        if select_num < 400:
            MCtimes = 20
        else:
            MCtimes = 5
        MCtimes = 1
        # Monte Carlo Sampling - MCTimes is the number of times that we sample
        for inner_i in range(MCtimes):
            grad_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, name) in enumerate(eval_data_loader):

                # break if 50 images (batches) done
                if cnt > 1 and args.debug:
                    break
                elif cnt > 200:
                    break

                data_time.update(time.time() - end)

                if torch.cuda.is_available():
                    image_var = Variable(image.cuda(), requires_grad=True)
                else:
                    image_var = Variable(image, requires_grad=True)

                # print("__shape of image var__", image_var.shape) # [1,3,1024,2048]
                final = model(image_var)[0]
                # print("__shape of final__", final.shape) # [1, 19, 1024,2048]
                _, pred = torch.max(final, 1)
                # print("__shape of pred__", pred.shape)  # [1,1024,2048]

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
                # print('label', label)
                label = label.long()
                if GPU_flag:
                    # image.cuda()
                    # image_var.cuda() # BUG: too late
                    mask = mask.cuda()
                    mask_target = mask_target.cuda()
                    label = label.cuda()

                target, mask = Variable(label), Variable(mask)
                loss = cross_entropy2d(final * mask, target * mask_target, size_average=False)
                loss.backward()

                data_grad = image_var.grad
                np_data_grad = data_grad.cpu().numpy()
                # print(np_data_grad.shape)
                L2_grad_norm = np.linalg.norm(np_data_grad) / select_num # the 1/M \sum_M \partial{Loss_i}/\partial{input}
                grad_sum += L2_grad_norm
                # increment the batch # counter
                cnt += 1

                pred = pred.cpu().data.numpy()
                batch_time.update(time.time() - end)
                end = time.time()


            grad_avg = grad_sum / cnt # Represents the gradient average for batch. cnt is the number of samples in a batch.
            grad_sample_avg_sum += grad_avg # For each sampling this is the sum of avg gradients in that sample.

        grad_sample_avg_sum /= MCtimes

        result_list.append(grad_sample_avg_sum)

        print(select_num, 'middle result', result_list)
        np.save('{}_{}_graph_more.npy'.format(args.dataset, args.arch), result_list)

    print('Final', result_list)
    np.save('{}_{}_graph_more.npy'.format(args.dataset, args.arch), result_list)

    # not sure if has to be moved
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)

def test_drnseg_masked_attack(eval_data_loader, model, num_classes,
                         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None):
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
    # select_num_list = [1] + [i * 4 for i in range(1, 100)] + [400 + i*200 for i in range(100)]
    select_num_list = [i * 500 for i in range(1, 15)] + [10000, 15000, 20000]
    # [5, 10, 50 , 100, 200] +
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
        # Monte Carlo Sampling - MCTimes is the number of times that we sample
        for inner_i in range(MCtimes):
            acc_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, name) in enumerate(eval_data_loader):
                print('iter', iter)

                # break if 50 images (batches) done
                if cnt > 1 and args.debug:
                    break
                elif cnt > 50:
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

                #TODO - get the things reqd for its arguments such as criteria and tasks
                info = get_info(args.dataset)
                criteria = cross_entropy2d
                adv_image_var = PGD_drnseg_masked_attack_city(image_var,label,attack_mask,model,criteria,
                                                             args.epsilon,args.steps,args.dataset,
                                                             args.step_size,info,args,using_noise=True)

                # print("__shape of image var__", image_var.shape) # [1,3,1024,2048]
                final = model(adv_image_var)[0]
                # print("__shape of final__", final.shape) # [1, 19, 1024,2048]
                _, pred = torch.max(final, 1)
                # print("__shape of pred__", pred.shape)  # [1,1024,2048]


                def accuracy_masked_attack(preds, label,mask):
                    valid_label = (label >= 0)*(label<=18)
                    valid = valid_label* mask.bool()
                    acc_sum = (valid * (preds == label)).sum()
                    valid_sum = valid.sum()
                    acc = float(acc_sum) / (valid_sum + 1e-10)
                    return acc

                acc = accuracy_masked_attack(pred,label,attack_mask)

                acc_sum += acc

                cnt += 1

                batch_time.update(time.time() - end)
                end = time.time()


            acc_batch_avg = acc_sum / cnt # Represents the gradient average for batch. cnt is the number of samples in a batch.
            acc_sample_avg_sum += acc_batch_avg # For each sampling this is the sum of avg gradients in that sample.

        acc_sample_avg_sum /= MCtimes

        result_list.append(acc_sample_avg_sum)

        print(select_num, 'middle result', result_list)

    print('Final', result_list)


def test_masked_accuracy_outdim(eval_data_loader, model, num_classes,
                         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None):
    """
    Evaluates the effect of increasing output dimension on the accuracy after attack.
    Monte Carlo sampling will be used and the result would be averaged.
    First choose the number of pixels to calculate the loss for (output dimension) --> select_num.
    For each select_num, we do the following MC_times(as Monte Carlo sampling):
        Attack the image, compare the number of pixels for which the correct class is calculated.
    Average all these appropriately


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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    # exit(0)

    if torch.cuda.is_available():
        GPU_flag = True
    else:
        GPU_flag = False

    # Number of pixels to be selected for masking - analogous to number of output dimensions. Only these many pixels will be considered to calculate the loss.
    # select_num_list = [1] + [i * 4 for i in range(1, 100)] + [400 + i*200 for i in range(100)]
    select_num_list = [i * 1000 for i in range(1,50)]

    result_list = []
    for select_num in select_num_list:
        print("********")
        print("selecting {} of output".format(select_num))
        import random
        acc_sample_avg_sum = 0
        if select_num < 400:
            MCtimes = 5
        else:
            MCtimes = 5
        # Monte Carlo Sampling - MCTimes is the number of times that we sample
        for inner_i in range(MCtimes):
            # grad_sum = 0
            acc_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, mask) in enumerate(eval_data_loader):

                # break if 50 images (batches) done
                if cnt > 1 and args.debug:
                    break
                elif cnt > 200:
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
                # attack_mask = attack_mask.float()
                #
                # mask_target = mask.long()
                # label = label.long()

                if GPU_flag:
                    # image.cuda()
                    # image_var.cuda() # BUG: too late
                    attack_mask = attack_mask.cuda()
                    # mask_target = mask_target.cuda()
                    # label = label.cuda()
                attack_mask = Variable(attack_mask)
                # target = Variable(label)

                # Attack image
                from models.mtask_losses import get_losses_and_tasks
                criteria, tasks = get_losses_and_tasks(args)
                info = get_info(args.dataset)
                # print([(m,mask[m].type()) for m in mask.keys()])

                #shape of adv_image_var = [batch_size, 2, 336,680], is a float, image var was also a float
                adv_image_var = PGD_masked_attack_mtask_city(image_var,label,mask,attack_mask,model,criteria,tasks,
                                                             args.epsilon,args.steps,args.dataset,
                                                             args.step_size,info,args,using_noise=True)

                # Get prediction
                final = model(adv_image_var)
                # Final is a dict, final['segmentsemantic'] is a tensor float

                _, pred = torch.max(final['segmentsemantic'], 1)

                def accuracy_masked_attack(preds, label,mask):
                    valid_label = (label >= 0)*(label<=18)
                    valid = valid_label* mask.bool()
                    acc_sum = (valid * (preds == label)).sum()
                    valid_sum = valid.sum()
                    acc = float(acc_sum) / (valid_sum + 1e-10)
                    return acc

                acc = accuracy_masked_attack(pred,label['segmentsemantic'].squeeze(1),attack_mask)
                # acc = number of pixels with same class / total number of pixels

                acc_sum += acc

                cnt += 1 # Increment batch counter

                batch_time.update(time.time() - end)

                end = time.time()

            acc_batch_avg = acc_sum / cnt # Represents the accuracy average for batch. cnt is the number of samples in a batch.
            acc_sample_avg_sum += acc_batch_avg # For each sampling this is the sum of accuracy in that sample.

        acc_sample_avg_sum /= MCtimes

        result_list.append(acc_sample_avg_sum)

        print(select_num, 'middle result', result_list)
        # np.save('{}_{}_graph_more.npy'.format(args.dataset, args.arch), result_list)

    print('Final', result_list)
    # np.save('{}_{}_graph_more.npy'.format(args.dataset, args.arch), result_list)

    # not sure if has to be moved
    # if has_gt:  # val
    #     ious = per_class_iu(hist) * 100
    #     logger.info(' '.join('{:.03f}'.format(i) for i in ious))
    #     return round(np.nanmean(ious), 2)

# if __name__ == "__main__":
#

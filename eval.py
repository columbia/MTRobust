
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

from learning.attack import PGD_attack
from dataloaders.utils import decode_segmap

import data_transforms as transforms

try:
    from modules import batchnormsync
except ImportError:
    pass

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def eval_adv(eval_data_loader, model, num_classes, args=None, info=None, eval_score=None, calculate_specified_only=False,test_flag=False):

    print("___Entering Adversarial Validation validate_adv()___")

    score = AverageMeter()
    CELoss = AverageMeter()

    model.eval()
    hist = np.zeros((num_classes, num_classes))

    criterion = nn.NLLLoss(ignore_index=255)

    for iter, (image, label, name) in enumerate(eval_data_loader):

        if iter>50:
            break

        if calculate_specified_only:
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
            label = label.cuda()
            # clean_input = clean_input.cuda()

        # print("_____putting in model")

        input_var = Variable(input, requires_grad=False) #TODO: volatile is removed, use with torch.no_grad() if necessary
        final = model(input_var)[0]
        # print("_____out of model")
        _, pred = torch.max(final, 1)

        decoded_target = decode_segmap(label[0].cpu().data.numpy() if torch.cuda.is_available() else label[0].data.numpy(), args.dataset)
        decoded_target = np.moveaxis(decoded_target, 2, 0)
        decoded_class_prediction = decode_segmap(pred[0].cpu().data.numpy() if torch.cuda.is_available() else pred[0].data.numpy(), args.dataset)
        decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)


        if eval_score is not None:

            if calculate_specified_only:
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
                CELoss.update(cross_entropy2d(final,target_temp,size_average=False).item())
                label = target_temp.cpu().numpy()
                pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
                hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

            else:

                score.update(eval_score(final, label), input.size(0))
                CELoss.update(cross_entropy2d(final,label,size_average=False).item())

                label = label.numpy()
                pred = pred.cpu().numpy() if torch.cuda.is_available() else pred.numpy()
                hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

        end = time.time()

        freq_print = 1

        if iter % freq_print == 0:
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))

            logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

        if args.debug:
            break



    logger.info(' *****\n***OverAll***\n Score {top1.avg:.3f}'.format(top1=score))

    ious = per_class_iu(hist) * 100
    logger.info(' '.join('{:.03f}'.format(i) for i in ious))

    if test_flag:
        # Note: test_flag is for running experiments


        dict_advacc = {}
        # print("TYPES ",type(round(np.nanmean(ious), 2)),type(CELoss.avg),type(score.avg))
        dict_advacc['segmentsemantic'] = {
            "iou": round(np.nanmean(ious), 2),
            "loss":CELoss.avg.item(),
            "seg_acc": score.avg
        }
        return dict_advacc

    print(' *****\n***OverAll***\n Score {top1.avg:.3f}'.format(top1=score))
    print('mIoU', np.nanmean(ious))
    return round(np.nanmean(ious), 2)




def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    # model specific
    model_arch = args.arch
    if (model_arch.startswith('drn')):
        single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                              pretrained=False)
    elif (model_arch.startswith('fcn32')):
        # define the architecture for FCN.
        single_model = FCN32s(args.classes)
    else:
        single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                              pretrained=False)  # Replace with some other model
        print("Architecture unidentifiable, please choose between : fcn32s, dnn_")

    model = torch.nn.DataParallel(single_model)
    print('loading model from path : ', args.pretrained)
    if '.tar' in args.pretrained:
        model_load = torch.load(args.pretrained)
        # print('model load', model_load.keys())
        print('model epoch', model_load['epoch'], 'precision', model_load['best_prec1'])
        model.load_state_dict(model_load['state_dict'])
    else:
        print(torch.load(args.pretrained).keys())
        model.load_state_dict(torch.load(args.pretrained))


    if torch.cuda.is_available():
        model.cuda()

    test_loader = get_loader(args, phase,out_name=True)
    info = get_info(args.dataset)

    cudnn.benchmark = True

    if args.adv_test:
        # if args.select_class:
        mAP = eval_adv(test_loader, model, args.classes, args=args, info=info, eval_score=accuracy,
                       calculate_specified_only=args.select_class)
        # from learning.validate import validate_adv_test
        # mAP = validate_adv_test(test_loader, model, args.classes, save_vis=True,
        #                has_gt=True, output_dir=None, downsize_scale=args.downsize_scale,
        #                args=args, info=info)

    elif args.select_class:
        test_selected_class_grad(test_loader, model, args.classes, args)

    else:
        mAP = test_mask_rand(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=None, downsize_scale=args.downsize_scale,
                   args=args)


def test_selected_class_grad(eval_data_loader, model, num_classes, args,test_flag=False):
    '''
    Calculates the gradient sum over the selected categories
    :param eval_data_loader:
    :param model:
    :param num_classes:
    :param args:
    :return:
    '''
    grad_sum = 0
    cnt = 0
    for i, (input, target, name) in enumerate(eval_data_loader):
        if i<50:
            continue
        cnt += 1
        if args.select_class:
            # For each class that we have to train, get a mask(mask_as_none) such that the ignored classes are 1 and the classes to train are 0
            for tt, each in enumerate(args.train_category):
                if tt == 0:
                    # print('target size', target.size())
                    mask_as_none = 1 - (torch.ones_like(target).long() * each == target).long()
                else:
                    mask_as_none = mask_as_none.long() * (1 - (torch.ones_like(target).long() * each == target).long())
            target = target * (1-mask_as_none.long()) + 255 * torch.ones_like(target).long() * mask_as_none.long() # puts 255 for the classes we are not training on and one for the pixels we are training on.

            # put the non selected as ignore class
            target = target.long()

        select_num = torch.sum(1 - mask_as_none.long()).item() # Number of pixels that we are training on ie which have 1 in (1-mask_as_none)
        # print('select num', select_num)

        # Now loss is calculated after forward pass. Input remains the same, but the target is now changed such that ones except for the selected class on which we are training, other pixels are marked as 255
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input, requires_grad=True) # TODO: volatile keyword is deprecated, so use with torch.no_grad():; see if necessary
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = cross_entropy2d(output, target_var, size_average=False)
        loss.backward()

        data_grad = input_var.grad

        np_data_grad = data_grad.cpu().numpy()
        # print(np_data_grad.shape)
        L2_grad_norm = np.linalg.norm(np_data_grad)

        # Divide the norm by the number of pixels that we are training. Somewhat representative of the gradient per pixel.
        grad_sum += L2_grad_norm / select_num

        if i%10==0:
            print('temp grad', grad_sum/cnt)
        if cnt==100:
            break
        if args.debug:
            break

    if test_flag:
        return grad_sum*1.0/cnt
        # loss = criterion(output, target_var)

    print('\n\n')
    print('-----**************---')
    print('model path', args.pretrained)
    print('Average of gradient', grad_sum / cnt)
    print('-----**************---')
    print('\n\n')

def test_mask_rand(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False, downsize_scale=1, args=None):
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

    select_num_list = [1] + [i * 4 for i in range(1, 30)]

    result_list = []
    for select_num in select_num_list:
        print("********")
        print("selecting {} of output".format(select_num))
        import random
        grad_sample_avg_sum = 0
        MCtimes = 20
        for inner_i in range(MCtimes):
            grad_sum = 0
            cnt = 0
            print("MC time {}".format(inner_i))
            for iter, (image, label, name) in enumerate(eval_data_loader):
                # break if 50 images (batches) done
                if cnt > 100:
                    break

                data_time.update(time.time() - end)

                # crop image to see if works
                im_height = image.size(2)
                im_width = image.size(3)
                downsize_scale = downsize_scale  # change with a flag
                if (downsize_scale != 1):
                    im_height_downsized = (int)(im_height / downsize_scale)
                    im_width_downsized = (int)(im_width / downsize_scale)

                    delta_height = (im_height - im_height_downsized) // 2
                    delta_width = (im_width - im_width_downsized) // 2
                    # print("Image sizes before and after downsize",im_height, im_width, im_height_downsized, im_width_downsized)
                    image = image[:, :, delta_height:im_height_downsized + delta_height,
                            delta_width:im_width_downsized + delta_width]
                    label = label[:, delta_height:im_height_downsized + delta_height,
                            delta_width:im_width_downsized + delta_width]

                # print('img lab', image.size(), label.size())

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

                # Build mask for image
                mask = np.zeros((image_var.size(2) * image_var.size(3)), dtype=np.uint8)
                for iii in range(select_num):
                    mask[selected[iii]] = 1
                mask = mask.reshape(1, 1, image_var.size(2), image_var.size(3))
                mask = torch.from_numpy(mask)
                mask = mask.float()
                mask_target = mask.long()
                label = label.long()
                if GPU_flag:
                    # image.cuda()
                    # image_var.cuda() # BUG: too late
                    mask = mask.cuda()
                    mask_target = mask_target.cuda()
                    label = label.cuda()

                target, mask = Variable(label), Variable(mask)
                loss = cross_entropy2d(final * mask, target * mask_target, size_average=False)  # TODO: here we calculate the sum
                # real gradient is mean, but it is hard to calculate mean here because of changing mask num
                loss.backward()

                data_grad = image_var.grad
                np_data_grad = data_grad.cpu().numpy()
                # print(np_data_grad.shape)
                L2_grad_norm = np.linalg.norm(np_data_grad)
                grad_sum += L2_grad_norm / select_num
                # increment the batch # counter
                cnt += 1
                batch_time.update(time.time() - end)
                end = time.time()

            grad_avg = grad_sum / cnt
            grad_sample_avg_sum += grad_avg

        grad_sample_avg_sum /= MCtimes
        result_list.append(grad_sample_avg_sum)

        print(select_num, 'middle result', result_list)
        np.save('{}_{}_graph.npy'.format(args.dataset, args.arch), result_list)

    print('Final', result_list)
    np.save('{}_{}_graph.npy'.format(args.dataset, args.arch), result_list)

    # not sure if has to be moved
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def run_test(dataset, model, model_path, step_size, step_num, select_class, train_category, adv_test, test_batch_size, args):
    config_file_path = "config/{}_{}_config.json".format(model, dataset)

    with open(config_file_path) as config_file:
        config = json.load(config_file)

        import socket
        if socket.gethostname() == "deep":
            data_dir = config['data-dir_deep']
            # pretrained = config['pretrained_deep']
            backup_output_dir = config['backup_output_dir_deep']
        elif socket.gethostname() == "amogh":
            data_dir = config['data-dir_amogh']
            # pretrained = config['pretrained_amogh']
            backup_output_dir = config['backup_output_dir_amogh']
        else:
            data_dir = config['data-dir']
            # pretrained = config['pretrained']
            backup_output_dir = config['backup_output_dir']


        list_dir = config['list-dir']
        classes = config['classes']
        crop_size = config['crop-size']
        step = config['step']
        arch = config['arch']
        batch_size = config['batch-size']
        epochs = config['epochs']
        lr = config['lr']
        lr_mode = config['lr-mode']
        momentum = config['momentum']
        weight_decay = config['weight-decay']

        workers = config['workers']
        phase = config['phase']
        random_scale = config['random-scale']
        random_rotate = config['random-rotate']
        downsize_scale = config['downsize_scale']
        base_size = config['base_size']

        args.reg_lambda = config["reg_lambda"]
        args.drop_ratio = config["drop_ratio"]
        args.MC_times = config["MC_times"]

        args.test_batch_size = test_batch_size

        args.pixel_scale = config['pixel_scale']
        args.steps = step_num
        args.epsilon = config['epsilon'] * 1.0 / args.pixel_scale
        args.step_size = step_size * 1.0 / args.pixel_scale
        args.print_freq = config['print_freq']

        print('attack scale {}  budget epsilon {} steps {} step size {}'.
              format(args.pixel_scale, args.epsilon, args.steps, args.step_size))

        args.arch = model
        args.pretrained = model_path

        args.select_class = select_class

        if select_class:
            args.train_category = train_category
            args.others_id = config['others_id']
            args.weight_mul = 1  #TODO:

            args.calculate_specified_only = True
            assert args.others_id not in args.train_category

        # Setting args from config file
        args.adv_test = adv_test
        args.dataset = dataset

        args.config = config
        args.data_dir = data_dir
        args.list_dir = list_dir
        args.classes = classes
        args.crop_size = crop_size
        args.step = step
        args.arch = arch
        args.batch_size = batch_size
        args.epochs = epochs
        args.lr = lr
        args.lr_mode = lr_mode
        args.momentum = momentum
        args.weight_decay = weight_decay
        args.workers = workers
        args.phase = phase
        args.random_scale = random_scale
        args.random_rotate = random_rotate
        args.downsize_scale = downsize_scale
        args.backup_output_dir = backup_output_dir  # To save the backup files corresponding to a training experiment.
        # print('output args.backup_output_dir', args.backup_output_dir)
        args.base_size = base_size
        assert classes > 0

        args.bn_sync = False

        # print(' '.join(sys.argv))
        # print(args)

        if args.bn_sync:
            drn.BatchNorm = batchnormsync.BatchNormSync

    test_seg(args)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.debug=False

    # run_test('cityscape', 'drn_d_22', '/mnt/md0/2019Fall/SegSaveLog/cityscape/drn22-cityscape-nat/savecheckpoint/model_best.pth.tar',
    #          2, 0, False, [], False, 1, args)

    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/model_zoo/DRN/drn_d_22_cityscapes.pth',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)

    # Grad Norm
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/drn22-cityscape-nat/savecheckpoint/model_best.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1, args=args)
    # #
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select-build/train_drn_d_22_cityscape_2019-09-28_16:38:52/savecheckpoint/model_best.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)




    # calibrated epoch 30 grad norm
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/drn22-cityscape-nat/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)
    #
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select-build/train_drn_d_22_cityscape_2019-09-28_16:38:52/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)

    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select_on_6_city/train_drn_d_22_cityscape_2019-10-11_03:09:14/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)

    # trained on 12 tasks [2, 13, 14,3, 10, 11, 12, 17, 7, 4, 5, 9]
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select_class-12/select_on_12_city/train_drn_d_22_cityscape_2019-10-11_15:34:50/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=2, step_num=0, select_class=True, train_category=[2], adv_test=False, test_batch_size=1,
    #          args=args)



    # Trained on all classes
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/drn22-cityscape-nat/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
    #          args=args)
    # #

    # Trained on only binary class
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select-build/train_drn_d_22_cityscape_2019-09-28_16:38:52/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
    #          args=args)

    # Trained on 6 tasks and negative class: [2, 13, 14,3, 10, 11]
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select_on_6_city/train_drn_d_22_cityscape_2019-10-11_03:09:14/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
    #          args=args)

    # Trained on 12 tasks and negative class: [2, 13, 14,3, 10, 11, 12, 17, 7, 4, 5, 9]
    # run_test('cityscape', 'drn_d_22',
    #          '/mnt/md0/2019Fall/SegSaveLog/cityscape/select_class-12/select_on_12_city/train_drn_d_22_cityscape_2019-10-11_15:34:50/savecheckpoint/checkpoint_030.pth.tar',
    #          step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
    #          args=args)

    run_test('cityscape', 'drn_d_22',
             '/mnt/md0/2019Fall/SegSaveLog/cityscape/category_rerun_converge/1_classes_city/train_drn_d_22_cityscape_2019-10-20_16:25:06/savecheckpoint/checkpoint_120.pth.tar',
             step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
             args=args)
    run_test('cityscape', 'drn_d_22',
             '/mnt/md0/2019Fall/SegSaveLog/cityscape/category_rerun_converge/3_classes_city/train_drn_d_22_cityscape_2019-10-19_15:47:57/savecheckpoint/checkpoint_120.pth.tar',
             step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
             args=args)
    run_test('cityscape', 'drn_d_22',
            '/mnt/md0/2019Fall/SegSaveLog/cityscape/category_rerun_converge/6_classes_city/train_drn_d_22_cityscape_2019-10-18_01:55:21/savecheckpoint/checkpoint_120.pth.tar',
             step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
             args=args)

    run_test('cityscape', 'drn_d_22',
             '/mnt/md0/2019Fall/SegSaveLog/cityscape/category_rerun_converge/all_classes_city/train_drn_d_22_cityscape_2019-10-16_23:24:01/savecheckpoint/checkpoint_120.pth.tar',
             step_size=4, step_num=5, select_class=True, train_category=[2], adv_test=True, test_batch_size=1,
             args=args)






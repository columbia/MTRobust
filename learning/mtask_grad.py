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

import logging

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mtask_forone_grad(val_loader, model, criterion, task_name, args, test_vis=False):
    grad_sum = 0
    cnt = 0
    model.eval()

    score = AverageMeter()

    print('task to be calculated gradients', task_name)

    for i, (input, target) in enumerate(val_loader):

        if torch.cuda.is_available():
            input = input.cuda()
            for keys, tar in target.items():
                target[keys] = tar.cuda()


        # input.requires_grad_()
        input_var = torch.autograd.Variable(input, requires_grad=True)
        # input.retain_grad()

        output = model(input_var)

        first_loss = None
        loss_dict = {}
        for c_name, criterion_fun in criterion.items():
            if first_loss is None:
                first_loss = c_name
                # print('l output target', output)
                # print('ratget', target)
                loss_dict[c_name] = criterion_fun(output, target)
                # print('caname', c_name, loss_dict[c_name])
            else:
                loss_dict[c_name] = criterion_fun(output[c_name], target[c_name])

        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = loss_dict[each]
            else:
                grad_total_loss = grad_total_loss + loss_dict[each]

        # grad_total_loss = loss_dict['segmentsemantic'] + loss_dict['depth_zbuffer']
        grad_total_loss.backward()
        # print('deug val in grad in bugger grad', input_var.grad)  # Interesting, here we also able to get the grad

        if test_vis:
            from learning.utils_learn import accuracy
            score.update(accuracy(output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))


        # TODO: shit, the following code could not calculate the grad even if I specify. For unknown reason. drive me high fever
        #
        # first = True
        # for c_name, criterion_fun in criterion.items():
        #     # print('if in',c_name, task_name)
        #
        #     if c_name in task_name:
        #         print('get one')
        #         # loss_calculate = criterion[c_name](output[c_name], target[c_name])
        #         loss_calculate = criterion_fun(output[c_name], target[c_name])
        #
        #
        #         # loss_fn = lambda x, y: torch.nn.functional.cross_entropy(x.float(), y.long().squeeze(dim=1), ignore_index=0,
        #         #                                   reduction='mean')
        #         # loss_calculate = loss_fn(output[c_name], target[c_name].float())
        #
        #
        #         # o2 = criterion[c_name](output[c_name], target[c_name])
        #         # import pdb; pdb.set_trace()
        #         # loss_calculate = torch.mean(output[c_name] - target[c_name].float())
        #         if first:
        #             total_loss = loss_calculate
        #             first = False
        #
        #         else:
        #             total_loss = total_loss + loss_calculate  #TODO: vikram told me cannot be += here, because grad will override
        #
        #
        # input.retain_grad()
        # total_loss.backward()
        #
        # import pdb; pdb.set_trace()

        # print(input_var)
        # print(input_var.grad)

        data_grad = input_var.grad
        # print('data grad', data_grad)
        np_data_grad = data_grad.cpu().numpy()
        L2_grad_norm = np.linalg.norm(np_data_grad)
        grad_sum += L2_grad_norm
        # increment the batch # counter
        cnt += 1

        if args.debug:
            if cnt>200:
                break

    if test_vis:
        print('Clean Acc for Seg: {}'.format(score.avg))
    print('Vulnerability in Grad Norm')
    print("average grad for task {} :".format(task_name), grad_sum * 1.0 /cnt)

    return grad_sum * 1.0 /cnt



from learning.attack import PGD_attack_mtask, PGD_attack_mtask_L2, PGD_attack_mtask_city
from learning.utils_learn import accuracy
def mtask_forone_advacc(val_loader, model, criterion, task_name, args, info, epoch=0, writer=None,
                        comet=None, test_flag=False, test_vis=False, norm='Linf'):
    """
    NOTE: test_flag is for the case when we are testing for multiple models, need to return something to be able to plot and analyse
    """
    assert len(task_name) > 0
    avg_losses = {}
    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))

    for c_name, criterion_fun in criterion.items():
        avg_losses[c_name] = AverageMeter()

    seg_accuracy = AverageMeter()
    seg_clean_accuracy = AverageMeter()

    model.eval() # this is super important for correct including the batchnorm

    print("using norm type", norm)

    for i, (input, target, mask) in enumerate(val_loader):

        if test_vis:
            clean_output = model(Variable(input.cuda(), requires_grad=False))
            seg_clean_accuracy.update(accuracy(clean_output['segmentsemantic'], target['segmentsemantic'].long().cuda()),
                                      input.size(0))
        if args.steps == 0 or args.step_size == 0:
            args.epsilon = 0
        if norm == 'Linf':
            if args.dataset == 'taskonomy':
                adv_img = PGD_attack_mtask(input, target, mask, model, criterion, task_name, args.epsilon, args.steps, args.dataset,
                                 args.step_size, info, args, using_noise=True)
            elif args.dataset == 'cityscape':
                adv_img = PGD_attack_mtask_city(input, target, mask, model, criterion, task_name, args.epsilon, args.steps,
                                           args.dataset,
                                           args.step_size, info, args, using_noise=True)

        elif norm == 'l2':
            adv_img = PGD_attack_mtask_L2(input, target, mask, model, criterion, task_name, args.epsilon, args.steps,
                                       args.dataset,
                                       args.step_size)
        # image_var = Variable(adv_img.data, requires_grad=False)
        image_var = adv_img.data
        # image_var = input
        if torch.cuda.is_available():
            image_var = image_var.cuda()
            for keys, m in mask.items():
                mask[keys] = m.cuda()
            for keys, tar in target.items():
                target[keys] = tar.cuda()

        # print("diff", torch.sum(torch.abs(raw_input-image_var)))
        with torch.no_grad():
            output = model(image_var)

        sum_loss = None
        loss_dict = {}

        for c_name, criterion_fun in criterion.items():
            this_loss = criterion_fun(output[c_name].float(), target[c_name],
                                      mask[c_name])
            if sum_loss is None:
                sum_loss = this_loss
            else:
                sum_loss = sum_loss + this_loss

            loss_dict[c_name] = this_loss
            avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))

        if 'segmentsemantic' in criterion.keys():
            # this is accuracy for segmentation
            seg_accuracy.update(accuracy(output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))

            #TODO: also mIOU here
            class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
            target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target['segmentsemantic'].data.numpy()
            class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

            if i % 500 == 0:
                class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                # print(target['segmentsemantic'].shape)
                decoded_target = decode_segmap(
                    target['segmentsemantic'][0][0].cpu().data.numpy() if torch.cuda.is_available() else
                    target['segmentsemantic'][0][0].data.numpy(),
                    args.dataset)
                decoded_target = np.moveaxis(decoded_target, 2, 0)
                decoded_class_prediction = decode_segmap(
                    class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else class_prediction[
                        0].data.numpy(), args.dataset)
                decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)
                if not test_flag:
                    writer.add_image('Val/image clean ', back_transform(input, info)[0])
                    writer.add_image('Val/image adv ', back_transform(adv_img, info)[0])
                    writer.add_image('Val/image gt for adv ', decoded_target)
                    writer.add_image('Val/image adv prediction ', decoded_class_prediction)

                    # if comet is not None: comet.log_image(back_transform(input, info)[0].cpu(),     name='Val/image clean ',          image_channels='first')
                    # if comet is not None: comet.log_image(back_transform(adv_img, info)[0].cpu(),   name='Val/image adv ',            image_channels='first')
                    # if comet is not None: comet.log_image(decoded_target,                           name='Val/image gt for adv ',     image_channels='first')
                    # if comet is not None: comet.log_image(decoded_class_prediction,                 name='Val/image adv prediction ', image_channels='first')

        if 'segmentsemantic' in criterion.keys():
            # this is accuracy for segmentation
            seg_accuracy.update(accuracy(output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))

            #TODO: also mIOU here
            class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
            target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target['segmentsemantic'].data.numpy()
            class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

        if args.debug:
            if i>1:
                break

    if test_vis:
        print("clean seg accuracy: {}".format(seg_clean_accuracy.avg))

    str_attack_result = ''
    str_not_attacked_task_result = ''
    for keys, loss_term in criterion.items():
        if keys in task_name:
            str_attack_result += 'Attacked Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])
        else:
            str_not_attacked_task_result += 'Not att Task Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])

    # Tensorboard logger
    if not test_flag:
        for keys, _ in criterion.items():
            if keys in task_name:
                    writer.add_scalar('Val Adv Attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg, epoch)
                    if comet is not None: comet.log_metric('Val Adv Attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg)
            else:
                    writer.add_scalar('Val Adv  not attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg)
                    if comet is not None: comet.log_metric('Val Adv  not attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg)

    if 'segmentsemantic' in criterion.keys() or 'segmentsemantic' in criterion.keys():
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        mIoU = round(np.nanmean(ious), 2)

        str_attack_result += '\n Segment Score ({score.avg:.3f}) \t'.format(score=seg_accuracy)
        str_attack_result += ' Segment ===> mAP {}\n'.format(mIoU)

        if comet is not None: comet.log_metric('segmentsemantic Attacked IOU',    mIoU)
        if comet is not None: comet.log_metric('segmentsemantic Attacked Score',  seg_accuracy)

    print('clean task')
    print(str_not_attacked_task_result)
    if test_flag:

        dict_losses = {}
        for key, loss_term in criterion.items():
            dict_losses[key] = avg_losses[key].avg
            # print(str_attack_result, "\nnew", avg_losses[keys].avg, "\n")
        if 'segmentsemantic' in criterion.keys():
            dict_losses['segmentsemantic'] = {'iou'    : mIoU,
                                              'loss'   : avg_losses['segmentsemantic'].avg,
                                              'seg_acc': seg_accuracy.avg}

        print("These losses are returned", dict_losses)
        #Compute the dictionary of losses that we want. Desired: {'segmentsemantic:[mIoU, cel],'keypoints2d':acc,'}
        return dict_losses


def mtask_test_all(val_loader, model, criterion, task_name, all_task_name_list, args, info, writer=None, epoch=0,
                   test_flag=False, test_vis=False):
    """
    task name: is not sorted here, so can be rigorously define the sequence of tasks
    all_task_name_list: make the task under attack first.
        NOTE: test_flag is for the case when we are testing for multiple models, need to return something to be able to plot and analyse
        """
    assert len(task_name) > 0
    avg_losses = {}
    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))
    num_of_tasks = len(all_task_name_list)

    for c_name, criterion_fun in criterion.items():
        avg_losses[c_name] = AverageMeter()

    seg_accuracy = AverageMeter()
    seg_clean_accuracy = AverageMeter()

    matrix_cos_all = np.zeros((num_of_tasks, num_of_tasks))
    matrix_cos = np.zeros((num_of_tasks, num_of_tasks))

    grad_norm_list_all = np.zeros((num_of_tasks))
    grad_norm_list = np.zeros((num_of_tasks))
    grad_norm_joint_all = 0

    model.eval()  # this is super important for correct including the batchnorm

    for i, (input, target, mask) in enumerate(val_loader):

        if test_vis:
            clean_output = model(Variable(input.cuda(), requires_grad=False))
            seg_clean_accuracy.update(
                accuracy(clean_output['segmentsemantic'], target['segmentsemantic'].long().cuda()),
                input.size(0))

        adv_img = PGD_attack_mtask(input, target, mask, model, criterion, task_name, args.epsilon, args.steps,
                                   args.dataset,
                                   args.step_size, info, args, using_noise=True)
        # image_var = Variable(adv_img.data, requires_grad=False)
        image_var = adv_img.data

        # print("diff", torch.sum(torch.abs(raw_input-image_var)))
        grad_list = []

        if torch.cuda.is_available():
            for keys, tar in mask.items():
                mask[keys] = tar.cuda()
            input = input.cuda()
            for keys, tar in target.items():
                target[keys] = tar.cuda()

        total_grad = None
        for jj, each in enumerate(all_task_name_list):
            input_var = torch.autograd.Variable(input, requires_grad=True)
            output = model(input_var)
            # total_loss = criterion['Loss'](output, target)
            loss_task = criterion[each](output[each], target[each], mask[each])
            loss_task.backward()

            grad = input_var.grad.cpu().numpy()
            grad_norm_list[jj] = np.linalg.norm(grad)

            grad_normalized = grad / np.linalg.norm(grad)

            grad_list.append(grad_normalized)

        input_var = torch.autograd.Variable(input, requires_grad=True)
        output = model(input_var)
        total_loss = 0
        for jj, each in enumerate(all_task_name_list):
            total_loss = total_loss + criterion[each](output[each], target[each], mask[each])

        total_loss.backward()
        total_grad = input_var.grad.cpu().numpy()

        grad_norm_joint_all += np.linalg.norm(total_grad)
        total_grad = total_grad / np.linalg.norm(
            total_grad)  # TODO: this is crucial for preventing GPU memory leak,

        for row in range(num_of_tasks):
            for column in range(num_of_tasks):
                if row < column:
                    matrix_cos[row, column] = np.sum(np.multiply(grad_list[row], grad_list[column]))
                elif row == column:
                    matrix_cos[row, row] = np.sum(np.multiply(grad_list[row], total_grad))

        matrix_cos_all = matrix_cos_all + matrix_cos
        grad_norm_list_all = grad_norm_list_all + grad_norm_list

        with torch.no_grad():
            output = model(image_var)

        # first_loss = None
        # loss_dict = {}
        # for c_name, criterion_fun in criterion.items():
        #     # if c_name in task_name:
        #     if first_loss is None:
        #         first_loss = c_name
        #         loss_dict[c_name] = criterion_fun(output, target)
        #     else:
        #         loss_dict[c_name] = criterion_fun(output[c_name], target[c_name])
        #     avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))

        for c_name, criterion_fun in criterion.items():
            avg_losses[c_name].update(criterion_fun(output[c_name], target[c_name], mask[c_name]).data.item(), input.size(0))

        if 'segmentsemantic' in criterion.keys():
            # this is accuracy for segmentation
            seg_accuracy.update(accuracy(output['segmentsemantic'], target['segmentsemantic'].long()),
                                input.size(0))

            # TODO: also mIOU here
            class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
            target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target[
                'segmentsemantic'].data.numpy()
            class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

            if i % 500 == 0:
                class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                # print(target['segmentsemantic'].shape)
                decoded_target = decode_segmap(
                    target['segmentsemantic'][0][0].cpu().data.numpy() if torch.cuda.is_available() else
                    target['segmentsemantic'][0][0].data.numpy(),
                    args.dataset)
                decoded_target = np.moveaxis(decoded_target, 2, 0)
                decoded_class_prediction = decode_segmap(
                    class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else class_prediction[
                        0].data.numpy(), args.dataset)
                decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)
                if not test_flag:
                    writer.add_image('Val/image clean ', back_transform(input, info)[0])
                    writer.add_image('Val/image adv ', back_transform(adv_img, info)[0])
                    writer.add_image('Val/image gt for adv ', decoded_target)
                    writer.add_image('Val/image adv prediction ', decoded_class_prediction)

        if args.debug:
            if i > 1:
                break

    if test_vis:
        print("clean seg accuracy: {}".format(seg_clean_accuracy.avg))

    str_attack_result = ''
    str_not_attacked_task_result = ''
    for keys, loss_term in criterion.items():
        if keys in task_name:
            str_attack_result += 'Attacked Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys,
                                                                                              loss=avg_losses[keys])
        else:
            str_not_attacked_task_result += 'Not att Task Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys,
                                                                                                             loss=
                                                                                                             avg_losses[
                                                                                                                 keys])

    # Tensorboard logger
    if not test_flag:
        for keys, _ in criterion.items():
            if keys in task_name:
                writer.add_scalar('Val Adv Attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg, epoch)
            else:
                writer.add_scalar('Val Adv  not attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg,
                                  epoch)

    if 'segmentsemantic' in criterion.keys():
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        mIoU = round(np.nanmean(ious), 2)

        str_attack_result += '\n Segment Score ({score.avg:.3f}) \t'.format(score=seg_accuracy)
        str_attack_result += ' Segment ===> mAP {}\n'.format(mIoU)

    print('clean task')
    print(str_not_attacked_task_result)
    if test_flag:

        dict_losses = {}
        for key, loss_term in criterion.items():
            dict_losses[key] = avg_losses[key].avg
            # print(str_attack_result, "\nnew", avg_losses[keys].avg, "\n")
        if 'segmentsemantic' in criterion.keys():
            dict_losses['segmentsemantic'] = {'iou': mIoU,
                                               'loss': avg_losses['segmentsemantic'].avg,
                                               'seg_acc': seg_accuracy.avg}

        print("These losses are returned", dict_losses)
        # Compute the dictionary of losses that we want. Desired: {'segmentsemantic:[mIoU, cel],'keypoints2d':acc,'}
        return dict_losses, matrix_cos_all, grad_norm_joint_all, grad_norm_list_all
    # the matrix, here, task under attack is the first


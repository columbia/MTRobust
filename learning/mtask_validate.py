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

from learning.utils_learn import accuracy

def mtask_test_clean(val_loader, model, criterion, task_for_test, args=None, info=None,
                     use_existing_img=False, given_img_set=None):
    score = AverageMeter()
    avg_losses = {}
    # print('criter test', criterion)
    for c_name, criterion_fun in criterion.items():
        avg_losses[c_name] = AverageMeter()


    torch.cuda.empty_cache()
    model.eval()
    model.cuda()
    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))

    # print('given_img_set', given_img_set)

    with torch.no_grad():
        cur = 0
        for i, (input, target, mask) in enumerate(val_loader):
            batch_size = input.size(0)
            if use_existing_img:
                if cur+batch_size>given_img_set.shape[0]:
                    break
                input = torch.from_numpy(given_img_set[cur:cur+batch_size])
                cur += batch_size

            if torch.cuda.is_available():
                input = input.cuda()
                for keys, tar in target.items():
                    target[keys] = tar.cuda()
                for keys, m in mask.items():
                    mask[keys] = m.cuda()
            # print('input size', input.size())
            output = model(input)

            # first_loss = None
            # loss_dict = {}
            # for c_name, criterion_fun in criterion.items():
            #     if first_loss is None:
            #         first_loss = c_name
            #         loss_dict[c_name] = criterion_fun(output, target)
            #     else:
            #         loss_dict[c_name] = criterion_fun(output[c_name], target[c_name])
            #     avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))

            sum_loss = None
            loss_dict = {}

            for c_name, criterion_fun in criterion.items():
                # this_loss = criterion_fun(output[c_name].float(), target[c_name],
                #                           target['mask'] if 'mask' in target else None)
                this_loss = criterion_fun(output[c_name].float(), target[c_name], mask[c_name])
                if sum_loss is None:
                    sum_loss = this_loss
                else:
                    sum_loss = sum_loss + this_loss

                loss_dict[c_name] = this_loss
                avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))



            if 'segmentsemantic' in criterion.keys():
                # this is accuracy for segmentation
                score.update(accuracy(output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))

                #TODO: also mIOU here
                class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target['segmentsemantic'].data.numpy()
                class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
                hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

        dict_losses = {}
        for key, loss_term in criterion.items():
            dict_losses[key] = avg_losses[key].avg
        if 'segmentsemantic' in criterion.keys():
            ious = per_class_iu(hist) * 100
            logger.info(' '.join('{:.03f}'.format(i) for i in ious))
            mIoU = round(np.nanmean(ious), 2)
            dict_losses['segmentsemantic'] = {'iou': mIoU,
                                               'loss': avg_losses['segmentsemantic'].avg,
                                               'seg_acc': score.avg}
        print("These losses are returned", dict_losses)
        return dict_losses

def mtask_validate(val_loader, model, criteria, writer, comet=None, args=None, eval_score=None, print_freq=200, info=None, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    torch.cuda.empty_cache()

    # print('info', info)

    avg_losses = {}
    # print('criter test', criteria)
    for c_name, criterion_fun in criteria.items():
        avg_losses[c_name] = AverageMeter()

    print("___Entering Validation validate()___")

    # switch to evaluate mode
    model.eval()

    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))

    end = time.time()
    with torch.no_grad():
        for i, (input, target, mask) in enumerate(val_loader):
            if args.debug:
                print("nat validate size", input.size())

            if torch.cuda.is_available():
                input = input.cuda()
                for keys, tar in target.items():
                    target[keys] = tar.cuda()
                for keys, m in mask.items():
                    mask[keys] = m.cuda()

            input_var = torch.autograd.Variable(input) # TODO: volatile keyword is deprecated, so use with torch.no_grad():; see if necessary

            # compute output
            output = model(input_var)
            # loss = criteria(output, target_var)

            sum_loss = None
            loss_dict = {}

            for c_name, criterion_fun in criteria.items():
                this_loss = criterion_fun(output[c_name].float(), target[c_name], mask[c_name])
                if sum_loss is None:
                    sum_loss = this_loss
                else:
                    sum_loss = sum_loss + this_loss

                loss_dict[c_name] = this_loss
                avg_losses[c_name].update(loss_dict[c_name].data.item(), input.size(0))

            # loss_dict['segmentsemantic'].backward()
            # print('deug val in grad', input_var.grad)  # Interesting, here we also able to get the grad

            if eval_score is not None and 'segmentsemantic' in criteria.keys():
                # this is accuracy for segmentation
                score.update(eval_score(output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))

                #TODO: also mIOU here
                class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target['segmentsemantic'].data.numpy()
                class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
                hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.debug:
                print_freq = 10
            if i % print_freq == 0:

                str = 'Test: [{0}/{1}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time)

                for keys, loss_term in loss_dict.items():
                    str += 'Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])

                # Show TensorBoard visualisations for image labels and the model predictions.
                # Show clean image
                writer.add_image('Val/Image_Clean', back_transform(input_var, info)[0])
                # if comet is not None: comet.log_image(back_transform(input_var, info)[0].cpu(), name='Val/Image_Clean', image_channels='first')
                # Show targets and predictions for each task
                for task_name, loss in criteria.items():
                    # Only show for valid task names
                    print(task_name)
                    if task_name != "mask" and task_name != "rgb":
                        # Show tensorboard visualisations related to segmentation as we need to decode the labels to corresponding colors.
                        if task_name == "segmentsemantic":
                            class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                            decoded_target = decode_segmap(
                                target['segmentsemantic'][0][0].cpu().data.numpy() if torch.cuda.is_available() else target['segmentsemantic'][0][0].data.numpy(),
                                args.dataset)
                            image_label = np.moveaxis(decoded_target, 2, 0)
                            decoded_class_prediction = decode_segmap(
                                class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else
                                class_prediction[
                                    0].data.numpy(), args.dataset)
                            task_prediction = np.moveaxis(decoded_class_prediction, 2, 0)

                        elif task_name == 'autoencoder':
                            transformed_image_label = back_transform(target[task_name], info)
                            transformed_task_prediction = back_transform(output[task_name], info)
                            # image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else target[task_name][0].data.numpy()
                            image_label = transformed_image_label[
                                0].cpu().data.numpy() if torch.cuda.is_available() else transformed_image_label[
                                0].data.numpy()
                            task_prediction = transformed_task_prediction[
                                0].cpu().data.numpy() if torch.cuda.is_available() else transformed_task_prediction[
                                0].data.numpy()

                        elif task_name == 'depth_zbuffer':
                            image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else \
                            target[task_name][0].data.numpy()
                            task_prediction = output[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else \
                            output[task_name][0].data.numpy()

                        else:
                            image_label = target[task_name][0].cpu().data.numpy() if torch.cuda.is_available() else \
                            target[task_name][0].data.numpy()
                            task_prediction = output[task_name][
                                0].cpu().data.numpy() if torch.cuda.is_available() else output[task_name][
                                0].data.numpy()

                        if image_label.shape[0] != 2:
                            group_image_label_and_prediction = np.stack((image_label, task_prediction))
                            writer.add_images('Val/Image_label_and_prediction/{}'.format(task_name),
                                              group_image_label_and_prediction)
                            # if image_label.shape[0] == 1:       image_label     =   image_label.squeeze(0)
                            # if task_prediction.shape[0] == 1:   task_prediction =   task_prediction.squeeze(0)

                            # if comet is not None: comet.log_image(image_label,        name='Train/Image_label', image_channels='first')
                            # if comet is not None: comet.log_image(task_prediction,    name='Train/prediction',  image_channels='first')

                if 'segmentsemantic' in criteria.keys():
                    str += 'Score {score.val:.3f} ({score.avg:.3f})\n'.format(score=score)
                    str += '===> mAP {mAP:.3f}'.format(mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2))

                logger.info(str)

                # Visualise the images
                if args.debug and i==print_freq:
                    break

    # logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    # Tensorboard logger
    for keys, _ in criteria.items():
        writer.add_scalar('Val Clean/ Avg Loss {}'.format(keys), avg_losses[keys].avg, epoch)
        if comet is not None: comet.log_metric('Val Clean/ Avg Loss {}'.format(keys), avg_losses[keys].avg, step=epoch)
    if 'segmentsemantic' in criteria.keys():
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        mIoU = round(np.nanmean(ious), 2)

        writer.add_scalar('Val Clean/ Seg accuracy ',   score.avg,  epoch)
        writer.add_scalar('Val Clean/ Seg mIoU',        mIoU,       epoch)
        if comet is not None: comet.log_metric ('Val Clean/ Seg accuracy ',   score.avg,  step=epoch)
        if comet is not None: comet.log_metric ('Val Clean/ Seg mIoU',        mIoU,       step=epoch)

    # Print log info
    str = 'Test \n'
    for keys, _ in criteria.items():
        str += 'Test Loss: {}  ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])
    if 'segmentsemantic' in criteria.keys():
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        mIoU = round(np.nanmean(ious), 2)

        str += '\n Segment Score ({score.avg:.3f}) \t'.format(score=score)
        str += ' Segment ===> mAP {}\n'.format(mIoU)

    logger.info(str)
    #

    torch.cuda.empty_cache()

    return np.mean([l.avg for l in avg_losses.values()])

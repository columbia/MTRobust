from learning.attack import PGD_attack_mtask, PGD_attack_mtask_L2, PGD_attack_mtask_city
from learning.utils_learn import accuracy

import numpy as np
from learning.ensemble_attack import PGD_attack_ensemble_mtask
import torch
from learning.utils_learn import *


import logging

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mtask_ensemble_test(val_loader, model_ensemble, criterion_list, task_name, args, use_houdini=False):
    avg_losses = {}
    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))

    model_ensemble.eval()
    seg_accuracy = AverageMeter()

    for c_name in task_name:
        avg_losses[c_name] = AverageMeter()

    for i, (input, target, mask) in enumerate(val_loader):

        if args.debug:
            if i>2:
                break

        adv_img = PGD_attack_ensemble_mtask(input, target, mask, model_ensemble, criterion_list, task_name, args.epsilon,
                                            args.steps, args.step_size, using_noise=args.use_noise, momentum=args.momentum, use_houdini=use_houdini)

        image_var = adv_img.data

        if torch.cuda.is_available():
            image_var = image_var.cuda()
            for keys, m in mask.items():
                mask[keys] = m.cuda()
            for keys, tar in target.items():
                target[keys] = tar.cuda()

        with torch.no_grad():
            output = model_ensemble(image_var)

        # Average over the individual predictions
        avg_output = {}
        # import pdb; pdb.set_trace()

        for each in task_name:
            each_output=None
            cnt=0
            for sub_output, sub_criteria in zip(output, criterion_list):
                if each_output is None:
                    each_output = sub_output[each]
                else:
                    each_output = each_output + sub_output[each] # Soft ensemble
                cnt += 1.
            avg_output[each] = each_output/cnt

            # overall test result
            criteria = criterion_list[0]
            loss = criteria[each](avg_output[each], target[each], mask[each])
            avg_losses[each].update(loss.data.item(), input.size(0))

        if 'segmentsemantic' in task_name:
            seg_accuracy.update(accuracy(avg_output['segmentsemantic'], target['segmentsemantic'].long()), input.size(0))
            class_prediction = torch.argmax(avg_output['segmentsemantic'], dim=1)
            target_seg = target['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target[
                'segmentsemantic'].data.numpy()
            class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

    str_attack_result = ''
    str_not_attacked_task_result = ''
    for keys in task_name:
        if keys in task_name:
            str_attack_result += 'Attacked Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys,
                                                                                              loss=avg_losses[keys])

    if 'segmentsemantic' in task_name or 'segment_semantic' in task_name:
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        mIoU = round(np.nanmean(ious), 2)

        str_attack_result += '\n Segment Score ({score.avg:.3f}) \t'.format(score=seg_accuracy)
        str_attack_result += ' Segment ===> mAP {}\n'.format(mIoU)

    dict_losses = {}
    for key in task_name:
        dict_losses[key] = avg_losses[key].avg
        # print(str_attack_result, "\nnew", avg_losses[keys].avg, "\n")
    if 'segmentsemantic' in task_name:
        dict_losses['segmentsemantic'] = {'iou': mIoU,
                                          'loss': avg_losses['segmentsemantic'].avg,
                                          'seg_acc': seg_accuracy.avg}

    # Compute the dictionary of losses that we want. Desired: {'segmentsemantic:[mIoU, cel],'keypoints2d':acc,'}
    return dict_losses















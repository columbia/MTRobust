from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor
from learning.utils_learn import *

class Houdini(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_pred, Y, task_loss, ignore_index=255):

        max_preds, max_inds = Y_pred.max(axis=1)

        mask            = (Y != ignore_index)
        Y               = torch.where(mask, Y, torch.zeros_like(Y).to(Y.device))
        true_preds      = torch.gather(Y_pred, 1, Y).squeeze(1)

        normal_dist     = torch.distributions.Normal(0.0, 1.0)
        probs           = 1.0 - normal_dist.cdf(true_preds - max_preds)
        loss            = torch.sum(probs * task_loss.squeeze(1) * mask.squeeze(1)) / torch.sum(mask.float())

        ctx.save_for_backward(Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss = ctx.saved_tensors

        C = 1./math.sqrt(2 * math.pi)

        temp        = C * torch.exp(-1.0 * (torch.abs(true_preds - max_preds) ** 2) / 2.0) * task_loss.squeeze(1) * mask.squeeze(1)
        grad_input  = torch.zeros_like(Y_pred).to(Y_pred.device)

        grad_input.scatter_(1, max_inds.unsqueeze(1), temp.unsqueeze(1))
        grad_input.scatter_(1, Y, -1.0 * temp.unsqueeze(1))

        grad_input          /= torch.sum(mask.float())

        return (grad_output * grad_input, None, None, None)

def custom_fast_hist(pred, label, n):
    k           = (label >= 0) & (label < n)
    label[~k]   = n
    temp        =  np.apply_along_axis(np.bincount, 1, n * label.astype(int) + pred, minlength=(n**2) + n)
    temp        = temp[:, : n**2]

    return temp.reshape(pred.shape[0], n, n)

def custom_per_class_iu(hist):
    diag = np.diagonal(hist, axis1=1, axis2=2)

    return diag / (hist.sum(2) + hist.sum(1) - diag + 1e-7)

def calc_iou(Y_pred, Y):
    num_classes         = Y_pred.shape[1]

    class_prediction    = torch.argmax(Y_pred, dim=1)
    class_prediction    = class_prediction.cpu().detach().numpy().reshape(class_prediction.shape[0], -1)
    target_seg          = Y.cpu().detach().numpy().reshape(Y.shape[0], -1)

    hist                = custom_fast_hist(class_prediction, target_seg, num_classes)
    ious                = custom_per_class_iu(hist) * 100
    mIoU                = np.nanmean(ious, axis=1)
    mIoU                = torch.Tensor(mIoU).to(Y_pred.device)

    # old_hist            = fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)
    # old_ious            = per_class_iu(old_hist) * 100
    # old_mIoU            = np.nanmean(old_ious)

    return mIoU

def PGD_attack_ensemble_mtask(x, y, mask, net, criterion_list, task_name, epsilon, steps, step_size, using_noise=True, momentum=False, use_houdini=False):
    net.eval()

    # tensor_std = get_torch_std(info)
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    rescale_term = 2./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term

    # print('epsilon', epsilon, epsilon / rescale_term)

    x_adv = x.clone()

    # print('x range', torch.min(x_adv), torch.max(x_adv))

    pert_upper = x_adv + epsilon
    pert_lower = x_adv - epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = -torch.ones_like(x_adv)  ####Big bug! used to be zero, totally wrong


    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)
    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()


    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        grad_total_loss = None
        if use_houdini: houdini = Houdini.apply

        for sub_output, sub_criteria in zip(h_adv, criterion_list):
            for each in task_name:
                if grad_total_loss is None:
                    if use_houdini:
                        task_loss       = calc_iou(sub_output[each].float(), y[each].long().squeeze(dim=1))
                        task_loss       = task_loss.reshape(-1, 1, 1, 1).repeat(1, 1, sub_output[each].shape[2], sub_output[each].shape[3])
                        h_loss          = houdini(sub_output[each].float(), y[each].long(), task_loss, 255)
                        grad_total_loss = h_loss
                    else:
                        grad_total_loss = sub_criteria[each](sub_output[each], y[each], mask[each])
                else:
                    if use_houdini:
                        task_loss       = calc_iou(sub_output[each], y[each].long().squeeze(dim=1))
                        h_loss          = houdini(sub_output[each].float(), y[each].long(), task_loss, 255)
                        grad_total_loss = grad_total_loss + h_loss
                    else:
                        grad_total_loss = grad_total_loss + sub_criteria[each](sub_output[each], y[each], mask[each])

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        if momentum:
            if i==0:
                grad = x_adv.grad
                old_grad = grad.clone()
            else:
                grad = old_grad * 0.5 + x_adv.grad
                old_grad = grad.clone()
        else:
            grad = x_adv.grad
        grad.sign_()
        x_adv = x_adv + step_size * grad #x_adv.grad #TODO: seems previously we may have bug?
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)

        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

        # sample = x_adv.data
        # im_rgb = np.moveaxis(sample[1].cpu().numpy().squeeze(), 0, 2)
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(im_rgb)
        # plt.show()

    return x_adv

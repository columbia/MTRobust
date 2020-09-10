from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform
import numpy as np
from learning.utils_learn import clamp_tensor

def get_torch_std(info):
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    return tensor_std


def PGD_attack_mtask_L2(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size):
    net.eval()

    # tensor_std = get_torch_std(info)
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    rescale_term = 2./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term #TODO: may need this if results not good

    x_adv = x.clone()
    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        x = x.cuda()
        x_adv = x_adv.cuda()
        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)

        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each]

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        grad = x_adv.grad

        # grad_normalized = grad / np.linalg.norm(grad)
        # print('epsilon', epsilon)
        x_adv = x_adv + grad * epsilon
        x_delta = x_adv - x
        x_delta_normalized = x_delta / torch.norm(x_delta, 2)

        x_adv = x + x_delta_normalized * epsilon

        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv



def PGD_attack_mtask(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
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

    pert_upper = x_adv + epsilon
    pert_lower = x_adv - epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = -torch.ones_like(x_adv)
    # lower_bound = torch.zeros_like(x_adv) #bug


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
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size * x_adv.grad
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)

        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    # sample =x_adv.data
    # im_rgb = np.moveaxis(sample[1].cpu().numpy().squeeze(), 0, 2)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(im_rgb)
    # plt.show()

    return x_adv



def PGD_attack_mtask_city(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    # std_array = np.asarray(info["std"])
    # tensor_std = torch.from_numpy(std_array)
    # tensor_std = tensor_std.unsqueeze(0)
    # tensor_std = tensor_std.unsqueeze(2)
    # tensor_std = tensor_std.unsqueeze(2).float()
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

        # # elif dataset == 'ade20k':
        # #     h_adv = net(x_adv,segSize = (256,256))
        #
        # # total_loss = 0
        # # for keys, loss_func in criterion:
        # #     if keys in task_name:
        # #         loss = loss_func(h_adv[keys], y[keys])
        # #         total_loss += loss
        #
        # first_loss = None
        # loss_dict = {}
        # for c_name, criterion_fun in criterion.items():
        #     if first_loss is None:
        #         first_loss = c_name
        #         # print('l output target', output)
        #         # print('ratget', target)
        #         loss_dict[c_name] = criterion_fun(h_adv, y)
        #         # print('caname', c_name, loss_dict[c_name])
        #     else:
        #         loss_dict[c_name] = criterion_fun(h_adv[c_name], y[c_name])
        #
        # grad_total_loss = None
        # for each in args.test_task_set:
        #     if grad_total_loss is None:
        #         grad_total_loss = loss_dict[each]
        #     else:
        #         grad_total_loss = grad_total_loss + loss_dict[each]



        # cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    # sample =x_adv.data
    # im_rgb = back_transform(sample, info)[0]
    # im_rgb = np.moveaxis(im_rgb.cpu().numpy().squeeze(), 0, 2)
    #
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(im_rgb)
    # plt.show()

    return x_adv

# def PGD_drnseg_masked_attack_city(image_var,label,mask,attack_mask,model,criteria,tasks,
#                                                              args.epsilon,args.steps,args.dataset,
#                                                              args.step_size,info,args,using_noise=True):
def PGD_drnseg_masked_attack_city(x, y, attack_mask, net, criterion, epsilon, steps, dataset, step_size, info, args, using_noise=True):
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

        # for keys, m in mask.items():
        #     mask[keys] = m.cuda()
        # for keys, tar in y.items():
        #     y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        #if i==10:
        #    print('PGD_drnseg_masked_attack_city attack step', i)
        h_adv = net(x_adv)  # dict{rep:float32,segmentasemantic:float32, depth_zbuffer:float32, reconstruct:float32}
        grad_total_loss = None
        # print("Task names ", task_name)
        # for each in task_name:
            # print("IN ",each)
            # if grad_total_loss is None:
                # print(each,y.keys(),h_adv[1])
                # print(h_adv)
        ignore_value = 255
                # print(mask[each].type(), attack_mask.type())
        attack_mask = attack_mask.long()
        # mask_each = mask[each]  # segmentsemantic is long and others are float.
        # mask_total = mask_each * attack_mask  # attack_mask is float, mask_total is float.
        # mask_total = mask_total.long()
                # print(each, (y[each] * mask_total).type())
                # print((ignore_value * (1-mask_total)).type()) # types(str, )
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


def PGD_masked_attack_mtask_city(x, y, mask, attack_mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    # std_array = np.asarray(info["std"])
    # tensor_std = torch.from_numpy(std_array)
    # tensor_std = tensor_std.unsqueeze(0)
    # tensor_std = tensor_std.unsqueeze(2)
    # tensor_std = tensor_std.unsqueeze(2).float()
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.
    
    ones_like_x_adv = torch.ones_like(x_adv) 
    
    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()
        tensor_std = tensor_std.cuda()

    ones_like_x_adv = torch.ones_like(x_adv)

    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()

    pert_epsilon = ones_like_x_adv * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv) # dict{rep:float32,segmentasemantic:float32, depth_zbuffer:float32, reconstruct:float32}
        grad_total_loss = None
        # print("Task names ", task_name)
        for each in task_name:
            # print("IN ",each)
            if grad_total_loss is None:
                # print(each,y.keys(),h_adv[1])
                # print(h_adv)
                ignore_value = 255
                # print(mask[each].type(), attack_mask.type())
                attack_mask = attack_mask.long()
                mask_each =  mask[each] #segmentsemantic is long and others are float.
                mask_total = mask_each * attack_mask # attack_mask is float, mask_total is float.
                mask_total = mask_total.long()
                # print(each, (y[each] * mask_total).type())
                # print((ignore_value * (1-mask_total)).type()) # types(str, )
                y[each] = y[each] * mask_total + ignore_value * (1-mask_total) # y is {auto:float,segsem:int64,deoth:float}
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each]*attack_mask)
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each]*attack_mask)

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv


def PGD_attack(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()



    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        # elif dataset == 'ade20k':
        #     h_adv = net(x_adv,segSize = (256,256))
        cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv




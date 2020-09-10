import torch
import collections


def cross_entropy_loss_mask(output, target, mask=None):
    if mask is None:
        raise TypeError("Mask is None")
    else:
        out =  torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1),
                                                 reduction='none')
        if len(mask.shape) == 4: mask = mask.squeeze(1)
        out *= mask.float()
        return out.mean()



def cross_entropy_loss(output, target, mask=None):
    return torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1),
                                             reduction='mean')

def soft_cross_entropy_loss(output, target, mask=None):
    log_likelihood = -torch.nn.functional.log_softmax(output.float(), dim=-1)
    return torch.mean(torch.mul(log_likelihood, target.float()))


def l1_loss(output, target, mask=None):
    return torch.nn.functional.l1_loss(output, target, reduction='mean')


def get_taskonomy_loss(losses):
    def taskonomy_loss(output, target):
        if 'mask' in target:
            mask = target['mask']
        else:
            mask = None

        sum_loss = None
        num = 0
        for n, t in target.items():
            if n in losses:
                o = output[n].float()
                if mask is None:
                    this_loss = losses[n](o, t, mask)
                else:
                    this_loss = losses[n](o, t, mask.float())

                num += 1
                if sum_loss is not None:
                    sum_loss = sum_loss + this_loss
                else:
                    sum_loss = this_loss

        return sum_loss  # /num # should not take average when using xception_taskonomy_new

    return taskonomy_loss

def l1_loss_mask(output, target, mask=None):
    if mask is None:
        raise TypeError("Mask is None")
    else:
        out = torch.nn.functional.l1_loss(output, target, reduction='none')
        out *= mask.float()
        return out.mean()


# DEFINE LOSSES FOR CITYSCAPE
# TODO: COMBINE LOSSES IF BETTER DESIGN
def segment_semantic_loss_cityscapes(output, target, mask=None):
    # Get from original Cityscapes
    loss = torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1),
                                      ignore_index=255,
                                      reduction='mean')
    return loss

def depth_loss_cityscapes(output, target, mask=None):

    loss = torch.nn.functional.l1_loss(output,target,reduction='none')
    mask = target < (2**13)
    loss *= mask.float()
    loss = loss.mean()
    return loss

def reconstruction_loss_cityscapes(output, target, mask=None):

    loss_fun = torch.nn.MSELoss()
    loss = loss_fun(output,target)
    #loss = loss.mean()
    return loss

def get_losses_and_tasks(args, customized_task_set=None):
    losses = {}
    criteria = {}

    if customized_task_set is None:
        task_set = args.task_set
    else:
        task_set = customized_task_set

    if args.dataset == 'taskonomy':
        taskonomy_tasks = []
        loss_map = {
            'autoencoder'           : l1_loss,
            'class_object'          : soft_cross_entropy_loss,
            'class_places'          : soft_cross_entropy_loss,
            'depth_euclidean'       : l1_loss_mask,
            'depth_zbuffer'         : l1_loss_mask,
            'depth'         : l1_loss_mask,
            'edge_occlusion'        : l1_loss_mask,
            'edge_texture'          : l1_loss,
            'keypoints2d'           : l1_loss,
            'keypoints3d'           : l1_loss_mask,
            'normal'                : l1_loss_mask,
            'principal_curvature'   : l1_loss_mask,
            'reshading'             : l1_loss_mask,
            'room_layout'           : soft_cross_entropy_loss,
            'segment_unsup25d'      : l1_loss,
            'segment_unsup2d'       : l1_loss,
            'segmentsemantic'       : cross_entropy_loss_mask,
            'segment_semantic'       : cross_entropy_loss_mask,
            'vanishing_point'       : l1_loss
        }
        for task in task_set:
            if task in loss_map:
                criteria[task]  = loss_map[task]
                taskonomy_tasks.append(task)
            else:
                print('unknown classes')
        return criteria, taskonomy_tasks


    elif args.dataset == 'cityscape':
        tasks = []

        if customized_task_set is None:
            task_set = args.task_set
        else:
            task_set = customized_task_set

        #TODO: Remove redundancy losses, criterion etc.
        for task_str in task_set:
            if 'segmentsemantic' in task_str:
                losses['segmentsemantic'] = segment_semantic_loss_cityscapes
                criteria['segmentsemantic'] = segment_semantic_loss_cityscapes
                tasks.append('segmentsemantic')

            elif 'segment_semantic' in task_str:
                losses['segment_semantic'] = segment_semantic_loss_cityscapes
                criteria['segment_semantic'] = segment_semantic_loss_cityscapes
                tasks.append('segment_semantic')

            elif 'depth_zbuffer' in task_str:
                losses['depth_zbuffer'] = depth_loss_cityscapes
                criteria['depth_zbuffer'] = depth_loss_cityscapes
                tasks.append('depth_zbuffer')

            elif 'depth' in task_str:
                losses['depth'] = depth_loss_cityscapes
                criteria['depth'] = depth_loss_cityscapes
                tasks.append('depth')

            elif 'autoencoder' in task_str:
                losses['autoencoder'] = reconstruction_loss_cityscapes
                criteria['autoencoder'] = reconstruction_loss_cityscapes
                tasks.append('autoencoder')
            elif 'reconstruct' in task_str:
                losses['reconstruct'] = reconstruction_loss_cityscapes
                criteria['reconstruct'] = reconstruction_loss_cityscapes
                tasks.append('reconstruct')

            else:
                print("UNKNOWN CLASS", "__{}__".format(task_str))

        # total_loss = get_taskonomy_loss(losses)

        return criteria, tasks  # TODO: need fix
        # return total_loss, losses, criteria, tasks  # TODO: need fix




import argparse
import json
import logging
import math
import os
import numpy as np
from os.path import exists, join, split
import threading
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

from dataloaders.datasets.pascal import VOCSegmentation
from dataloaders.datasets.coco import COCOSegmentation
from dataloaders.datasets.taskonomy import TaskonomyLoader
import data_transforms as transforms

def get_info(dataset):
    """ Returns dictionary with mean and std"""

    if dataset == 'voc':
        return VOCSegmentation.INFO

    elif dataset == 'coco':
        return COCOSegmentation.INFO

    elif dataset == 'cityscape':
        return SegList.INFO

    elif dataset == 'taskonomy':
        return TaskonomyLoader.INFO

def get_loader(args, split, out_name=False, customized_task_set=None):
    """Returns data loader depending on dataset and split"""
    dataset = args.dataset
    loader = None

    if customized_task_set is None:
        task_set = args.task_set
    else:
        task_set = customized_task_set

    if dataset == 'taskonomy':
        print('using taskonomy')
        if split == 'train':
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=True,
                                     threshold=1200,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True)

        if split == 'val':
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=False,
                                     threshold=1200,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True)

        if split == 'adv_val':
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=False,
                                     threshold=1200,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True)


    elif dataset == 'voc':
        if split == 'train':
            loader = torch.utils.data.DataLoader(
                VOCSegmentation(args=args, base_dir=args.data_dir, split='train'),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )
        elif split == 'val':
            loader = torch.utils.data.DataLoader(
                VOCSegmentation(args=args, base_dir=args.data_dir, split='val',out_name=out_name),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )
        elif split == 'adv_val':
            loader = torch.utils.data.DataLoader(
                VOCSegmentation(args=args, base_dir=args.data_dir, split='val',out_name=out_name),
                batch_size=1, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )

    elif dataset == 'coco':
        if split == 'train':
            loader = torch.utils.data.DataLoader(
                COCOSegmentation(args=args, base_dir=args.data_dir, split='train'),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )
        elif split == 'val':
            loader = torch.utils.data.DataLoader(
                COCOSegmentation(args=args, base_dir=args.data_dir, split='val',out_name=out_name),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )
        elif split == 'adv_val':
            loader = torch.utils.data.DataLoader(
                COCOSegmentation(args=args, base_dir=args.data_dir, split='val',out_name=out_name),
                batch_size=1, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )

    elif dataset == 'cityscape':
        data_dir = args.data_dir
        info = json.load(open(join(data_dir, 'info.json'), 'r'))
        normalize = transforms.Normalize(mean=info['mean'],
                                         std=info['std'])
        t = []
        if args.random_rotate > 0:
            t.append(transforms.RandomRotate(args.random_rotate))
        if args.random_scale > 0:
            t.append(transforms.RandomScale(args.random_scale))
        t.extend([transforms.RandomCrop(args.crop_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])

        task_set_present = hasattr(args,'task_set')
        if split == 'train':
            if task_set_present:
                print("\nCAUTION: THE DATALOADER IS FOR MULTITASK ON CITYSCAPE\n")
                loader = torch.utils.data.DataLoader(
                    SegDepthList(data_dir, 'train', transforms.Compose(t),
                                 list_dir=args.list_dir),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True
                )
            else:
                loader = torch.utils.data.DataLoader(
                    SegList(data_dir, 'train', transforms.Compose(t),
                            list_dir=args.list_dir),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True
                )
        elif split == 'val':
            if args.task_set != []:
                print("\nCAUTION: THE DATALOADER IS FOR MULTITASK ON CITYSCAPE\n")
                loader = torch.utils.data.DataLoader(
                    SegDepthList(data_dir, 'val', transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ]), list_dir=args.list_dir, out_name=out_name),
                    batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, drop_last=True
                )
            else:
                print("city test eval!")
                loader = torch.utils.data.DataLoader(
                    SegList(data_dir, 'val', transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), list_dir=args.list_dir, out_name=out_name),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
                )
        elif split == 'adv_val': # has batch size 1
            if task_set_present:
                print("\nCAUTION: THE DATALOADER IS FOR MULTITASK ON CITYSCAPE\n")
                loader = torch.utils.data.DataLoader(
                    SegDepthList(data_dir, 'val', transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]),
                    list_dir=args.list_dir,out_name=out_name),
                    batch_size=1, shuffle=False, num_workers=args.workers,
                    pin_memory=True, drop_last=True
                )
            else:
                loader = torch.utils.data.DataLoader(
                    SegList(data_dir, 'val', transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), list_dir=args.list_dir,out_name=out_name),
                batch_size=1, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True
            )

    return loader

class SegList(torch.utils.data.Dataset):

    INFO = {"mean": [0.29010095242892997,0.32808144844279574,0.28696394422942517],
            "std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]}

    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            # print("not none")
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        # print('data', data)
        # print('trans', self.transforms)
        data = list(self.transforms(*data))
        # print('data', data)
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        # print('final d', data)
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class SegDepthList(torch.utils.data.Dataset):
    """
    Dataloader for getting the multitask labels from Cityscapes.
    """
    INFO = {"mean": [0.29010095242892997,0.32808144844279574,0.28696394422942517],
            "std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]}

    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        im_file_path = self.image_list[index]

        targets = {}
        mask = {}

        # Prepare reconstruct label
        rgb_data = Image.open(join(self.data_dir, self.image_list[index]))

        # Prepare segmentation label
        segmentation_label = Image.open(join(self.data_dir, self.label_list[index]))

        # Prepare depth label
        disparity_img_path = join(self.data_dir, im_file_path.replace("leftImg8bit","disparity"))
        depth_img = (Image.open(disparity_img_path))
        # depth_img = torch.ByteTensor(torch.ByteStorage.from_buffer(depth_img.tobytes()))


        # np_depth_img = np.array(depth_img) # np array version of the pillow label read
        # np_depth_img_normalise = np_depth_img.astype(np.float) # np float version
        # np_depth_img_normalise = np.where(np_depth_img_normalise>0,(np_depth_img_normalise-1)/256.,0.) # Get the disparity using formula given on the website in 8bit representation
        # np_depth_img_normalise = np_depth_img_normalise.astype(np.uint8)
        # depth_img = Image.fromarray(np_depth_img_normalise) # Convert depth to PILLOW Image
        # depth_img = torch.FloatTensor(np_depth_img_normalise)
        # depth_img.show()
        # depth_label = self.transforms(*[depth_img,None])[0]
        rgb_data, segmentation_label, depth_label = self.transforms(*[rgb_data,segmentation_label,depth_img])
        # t = self.transforms(*t)
        depth_label = depth_label.unsqueeze(0)
        segmentation_label = segmentation_label.unsqueeze(0)

        depth_label = depth_label/1000.

        targets['autoencoder'] = rgb_data
        targets['reconstruct'] = rgb_data  #TODO: remove this after, this is for old version viz
        targets['segmentsemantic'] = segmentation_label
        targets['segment_semantic'] = segmentation_label #TODO: remove this after, this is for old version viz
        targets['depth_zbuffer'] = depth_label
        targets['depth'] = depth_label #TODO: remove  this after submission



        mask['autoencoder'] = torch.ones_like(rgb_data)
        mask['reconstruct'] = torch.ones_like(rgb_data)  #TODO: remove this after, this is for old version viz
        mask['segmentsemantic'] = torch.ones_like(segmentation_label)
        mask['segment_semantic'] = torch.ones_like(segmentation_label)  #TODO: remove this after, this is for old version viz
        mask['depth_zbuffer'] = torch.ones_like(depth_label)
        mask['depth'] = torch.ones_like(depth_label)  #TODO: remove  this after submission

        return rgb_data, targets, mask

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)



class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


if __name__ == "__main__":
    #Testing the dataloader
    data_dir = "/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset/"
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    # t.append(transforms.RandomRotate(0))
    # t.append(transforms.RandomScale(0))
    t.extend([transforms.RandomCrop(896),
              # transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    # loader = SegDepthList(data_dir="/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset/",
    loader=torch.utils.data.DataLoader(
        SegDepthList(data_dir, 'train', transforms.Compose(t),
                     list_dir=None),
        batch_size=1, shuffle=False, num_workers=1,
        pin_memory=True, drop_last=True
    )



    import matplotlib.pyplot as plt
    f,ax = plt.subplots(5)
    for i,(input,targets) in enumerate(loader):
        print("Inside Loader", i , input.shape)
        target_reconstruct = targets['autoencoder']
        target_segmentation = targets['segmentsemantic']
        target_depth = targets['depth_zbuffer']
        target_depth2 = target_depth[0].cpu().data.numpy() if torch.cuda.is_available() else target_depth[
            0].data.numpy()
        target_depth3 = np.where(target_depth2 > 0, (target_depth2 - 1) / 256., 0.)  # Get the disparity using formula given on the website in 8bit representation
        target_depth3 = target_depth3.astype(np.uint8)

        # task_prediction = task_prediction.astype(np.float)  # np float version
        print("target_reconstruct.shape", target_reconstruct.shape)
        print("target_segmentation.shape", target_segmentation.shape)
        print("target_depth.shape", target_depth.shape)
        ax[0].imshow(np.moveaxis(target_reconstruct[0].numpy(),0,2))
        ax[1].imshow(target_segmentation[0][0].numpy())
        ax[2].imshow(target_depth[0][0],cmap='gray')
        # ax[3].imshow(target_depth2[0])
        ax[4].imshow(target_depth3[0],cmap='gray')
        plt.show()


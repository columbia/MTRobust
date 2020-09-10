#########
# Script to downscale data, currently for Cityscape
# Usage -
#
#########

import os
import sys
import argparse
import numpy as np
import tqdm
import torch, torchvision
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import glob


def downscale(args):
    path_dataset = '/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset'
    path_disparity = os.path.join(path_dataset, 'disparity')
    path_rgb = os.path.join(path_dataset, 'leftImg8bit')
    path_seg_labels = os.path.join(path_dataset, 'gtFine')

    tensor_tr = torchvision.transforms.ToTensor()
    pil_tr = torchvision.transforms.ToPILImage()

def crop_image():
    def generate_dataset(path_dataset):
        list_paths = glob.glob(path_dataset + "/*/*/*/*.png")
        for im_path in tqdm.tqdm(list_paths):
            # Check path, generate output path string, read image, transform
            if "/test/" not in im_path:
                if "/disparity/" in im_path:
                    im = Image.open(im_path)

                else:
                    if "/gtFine/" in im_path:
                        im = Image.open(im_path)

                    elif "/leftImg8bit/" in im_path:
                        # output_path_label = im_path.replace("/leftImg8bit/", "/leftImg8bit_small/")
                        im = Image.open(im_path)

                # IF CROPPED IMAGE TO BE SAVED
                output_im = (im.crop((0, 0, 680, 336)))
                output_im.save(im_path)

def downscale_disparity_im(im):

    im_tensor = torch.IntTensor(np.array(im, dtype=np.int16))
    im_tensor = im_tensor.view((1, 1, im_tensor.shape[-2], im_tensor.shape[-1]))

    # Define filter
    filter_size = 3
    filter_tensor = torch.ones((filter_size, filter_size)) / (filter_size * filter_size)
    filter_tensor = filter_tensor.view((1, 1, filter_size, filter_size))

    # Convolve image with this filter
    filtered_im = torch.nn.functional.conv2d(im_tensor.float(), filter_tensor, stride=filter_size, padding=0)

    # Return the convolved image as a Pillow image
    filtered_im.type(torch.IntTensor)
    filtered_im_array = np.array(filtered_im[0][0], dtype=np.int16)
    filtered_im = Image.fromarray(filtered_im_array)
    return filtered_im

def downscale_segmentation_label(im,filter_size=3):

    # Change pillow image to tensor
    im_tensor = tensor_tr(im)
    im_tensor = im_tensor.view((1, 1, im_tensor.shape[-2], im_tensor.shape[-1]))

    # Define filter such that the center is 1, rest 0
    filter_size = filter_size
    filter_tensor = torch.zeros((filter_size, filter_size))
    filter_tensor[filter_size // 2][filter_size // 2] = 1.
    filter_tensor = filter_tensor.view((1, 1, filter_size, filter_size))

    # Convolve image with the filter
    filtered_im = torch.nn.functional.conv2d(im_tensor.float(), filter_tensor, stride=filter_size, padding=0)

    # Return image as a pillow image
    filtered_im = pil_tr(filtered_im[0])
    return filtered_im

def downscale_rgb_im(im,filter_size=3):
    im_height = im.size[0]
    im_width = im.size[1]
    output_im = im.resize((im_height, im_width), Image.ANTIALIAS)
    return output_im

def debug_downscale_disparity():
    sample_disparity_path = '/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset/disparity/train/bochum/bochum_000000_000600_disparity.png'
    sample_disparity_im = Image.open(sample_disparity_path)
    print(np.asarray(sample_disparity_im).min(), np.asarray(sample_disparity_im).max(),
          np.asarray(sample_disparity_im).mean(), )
    #     downscale_disparity_im(sample_disparity_im)\
    convolved_image = downscale_disparity_im(sample_disparity_im)
    #     print(sample_disparity_im.size)
    #     print(convolved_image.size())
    inp_arr = np.asarray(sample_disparity_im)
    out_arr = np.asarray(convolved_image)
    #     convolved_image.show()
    print("original array stats: ", np.min(inp_arr), np.max(inp_arr), np.mean(inp_arr))
    print("output array stats:", np.min(out_arr), np.max(out_arr), np.mean(out_arr))
    f, ax = plt.subplots(2, figsize=(8, 8))
    ax[0].imshow(np.asarray(convolved_image))
    ax[1].imshow(np.asarray(sample_disparity_im))

def debug_downscale_seg():
    sample_path = '/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset/gtFine/train/aachen/aachen_000000_000019_gtFine_trainIds.png'
    sample_im = Image.open(sample_path)
    #     print(np.array(sample_im).min(),np.array(sample_im).max())
    #     downscale_disparity_im(sample_disparity_im)\
    convolved_image = downscale_segmentation_label(sample_im)
    print(sample_im.size, convolved_image.size)
    #     print(convolved_image.size)
    #     convolved_image.show()
    conv_arr = np.array(convolved_image)
    #     conv_arr = np.expand_dims(np.expand_dims(conv_arr,0),0)
    #     print(np.array(conv_arr).min(),np.array(conv_arr).max())
    #     print(conv_arr.shape)
    sample_im = decode_segmap(np.asarray(sample_im), 'voc')
    conv_im = decode_segmap(conv_arr, 'voc')
    f, ax = plt.subplots(2)
    ax[0].imshow(conv_im)
    ax[1].imshow(sample_im)

def debug_downscale_rgb():
    sample_path = '/home/amogh/data/datasets/drn_data/DRN-move/cityscape_dataset/leftImg8bit/train/aachen/aachen_000009_000019_leftImg8bit.png'
    sample_im = Image.open(sample_path)
    #     downscale_disparity_im(sample_disparity_im)\
    convolved_image = downscale_rgb_im(sample_im)
    print(sample_im.size, convolved_image.size)
    #     print(convolved_image.size)
    #     convolved_image.show()
    f, ax = plt.subplots(2)
    ax[0].imshow(np.asarray(convolved_image))
    ax[1].imshow(np.asarray(sample_im))
    ax[0].set_title("convolved image")
    ax[1].set_title("sample_im")

def generate_dataset(path_dataset):
    list_paths = glob.glob(path_dataset + "/*/*/*/*.png")
    for im_path in tqdm.tqdm(list_paths):
        # Check path, generate output path string, read image, transform
        if "/test/" not in im_path:
            if "/disparity/" in im_path:
                continue
                output_path_label = im_path.replace("/disparity/", "/disparity_small/")
                im = Image.open(im_path)
                output_im = downscale_disparity_im(im)
            else:
                if "/gtFine/" in im_path:
                    #                 print(im_path)
                    output_path_label = im_path.replace("/gtFine/", "/gtFine_small/")
                    im = Image.open(im_path)
                    #                 im.show()
                    #                 break
                    if "trainIds" in im_path:
                        output_im = downscale_segmentation_label(im)
                    else:
                        print("plain")
                        output_im = im

                elif "/leftImg8bit/" in im_path:
                    output_path_label = im_path.replace("/leftImg8bit/", "/leftImg8bit_small/")
                    im = Image.open(im_path)
                    output_im = downscale_rgb_im(im)

                dir_name = os.path.dirname(output_path_label)
                #             print(os.path.exists(dir_name))
                if not os.path.exists(dir_name):
                    print("Directory does not exist: ", dir_name)
                    os.makedirs(dir_name)
                #         os.makedirs(osoutput_path_label)
                print(output_path_label)
                output_im.save(output_path_label)
                #     print(len(list_paths))

def parse_args():

    parser = argparse.ArgumentParser(description="Parse details to downscale Cityscape dataset")
    parser.add_argument('-d','--data-dir', default=None, help='Cityscape data directory')
    parser.add_argument('-f','--factor', type=int, default=3)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    downscale(args)

if __name__ == "__main__":
    main()
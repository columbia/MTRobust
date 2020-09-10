import os,sys
import PIL
# from PIL import Image
import tqdm
import glob
import numpy as np
import skimage
from   skimage.transform import resize
from   scipy.ndimage.interpolation import zoom

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)

    By kchen @ https://github.com/kchen92/joint-representation/blob/24b30ca6963d2ec99618af379c1e05e1f7026710/lib/data/input_pipeline_feed_dict.py
    """
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        interps = [PIL.Image.NEAREST, PIL.Image.BILINEAR]
        return (im.resize(new_dims, interps[interp_order]))

    if all( new_dims[i] == im.shape[i] for i in range( len( new_dims ) ) ):
        resized_im = im #return im.astype(np.float32)
    elif im.shape[-1] == 1 or im.shape[-1] == 3:
        resized_im = resize(im, new_dims, order=interp_order, preserve_range=True)
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    # resized_im = resized_im.astype(np.float32)
    return resized_im

def resize_dataset(path_dataset, list_tasks, new_dims, interp_order):
    list_paths = glob.glob(path_dataset + "/*/*.png")
    # For each image path in the dataset
    for im_path in tqdm.tqdm(list_paths):
        task_dir_name = os.path.basename(os.path.dirname(im_path))
        if task_dir_name in list_tasks:
            # list_tasks.remove(task_dir_name)
            im = PIL.Image.open(im_path)
            resized_image = resize_image(im, new_dims, interp_order)
            output_path = im_path.replace('taskonomy-sample-model-1', 'taskonomy-sample-model-1-small')
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                print("making dir: ", output_dir)
                os.makedirs(output_dir)
            save=True
            if save:
                resized_image.save(output_path)







if __name__ == "__main__":
    root_deep = "/mnt/md0/2019Fall/taskonomy/taskonomy-sample-model-1-master"
    root_amogh = "/home/amogh/data/taskonomy-sample-model-1"
    root = "/home/ubuntu/taskonomy-sample-model-1-master"
    list_tasks = ["rgb",
                  "depth_zbuffer",
                  "edge_texture",
                  "keypoints2d",
                  "normal",
                  "reshading",
                  "keypoints3d",
                  "depth_euclidean",
                  "edge_occlusion",
                  "principal_curvature",
                  ]
    new_dims = (256,256)
    # resize_dataset(root, list_tasks,new_dims,interp_order=1)

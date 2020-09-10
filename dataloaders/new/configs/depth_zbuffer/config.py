from __future__ import absolute_import, division, print_function

import functools
import numpy as np
import os
import sys


sys.path.insert( 1, os.path.realpath( '../../../../models' ) )
sys.path.insert( 1, os.path.realpath( '../../../../lib' ) )

import dataloaders.new.data.load_ops as load_ops
from   dataloaders.new.data.load_ops import mask_if_channel_ge
from   dataloaders.new.data.task_data_loading import load_and_specify_preprocessors


from dataloaders.new.configs.inp_config import get_inp_cfg


def get_cfg( nopause=False ):
    cfg = get_inp_cfg()
    # outputs
    cfg['target_num_channels'] = 1
    cfg['target_dim'] = (256, 256)  # (1024, 1024)
    cfg['target_domain_name'] = 'depth_zbuffer'
    cfg['target_preprocessing_fn'] = load_ops.resize_and_rescale_image_log
    cfg['target_preprocessing_fn_kwargs'] = {
        'new_dims': cfg['target_dim'],
        'offset': 1.,
        'normalizer': np.log( 2. ** 16.0 )
    }

    # masks
    cfg['mask_fn'] = mask_if_channel_ge # given target image as input
    cfg['mask_fn_kwargs'] = {
        'img': '<TARGET_IMG>',
        'channel_idx': 0,
        'threshhold': 64500, # roughly max value - 1000. This margin is for interpolation errors
        'broadcast_to_dim': cfg['target_num_channels']
    }

    #cfg['depth_mask'] = True

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors

    return cfg

from __future__ import absolute_import, division, print_function

import functools
import numpy as np
import os
import sys



import dataloaders.new.data.load_ops as load_ops
from   dataloaders.new.data.load_ops import mask_if_channel_le
from   dataloaders.new.data.task_data_loading import load_and_specify_preprocessors


from dataloaders.new.configs.inp_config import get_inp_cfg


def get_cfg( nopause=False ):
    cfg = get_inp_cfg()
    # outputs
    cfg['output_dim'] = (256, 256)
    cfg['num_pixels'] = 300
    cfg['only_target_discriminative'] = True
    cfg['target_num_channels'] = 64
    cfg['target_dim'] = (cfg['num_pixels'], 3)  # (1024, 1024)
    cfg['target_domain_name'] = 'segment_unsup25d'

    cfg['target_from_filenames'] = load_ops.segment_pixel_sample
    cfg['target_from_filenames_kwargs'] = {
        'new_dims': cfg['output_dim'],
        'num_pixels': cfg['num_pixels'],
        'domain': cfg['target_domain_name']
    }

    cfg['return_accuracy'] = False

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors

    return cfg

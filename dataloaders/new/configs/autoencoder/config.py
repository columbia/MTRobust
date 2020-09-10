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
    cfg['target_num_channels'] = 3
    cfg['target_dim'] = (256, 256)  # (1024, 1024)
    cfg['target_domain_name'] = 'rgb'
    cfg['target_preprocessing_fn'] = load_ops.resize_rescale_image
    cfg['target_preprocessing_fn_kwargs'] = {
        'new_dims': cfg['target_dim'],
        'new_scale': [-1, 1]
    }

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors

    return cfg

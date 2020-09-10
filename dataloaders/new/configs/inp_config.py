from __future__ import absolute_import, division, print_function

import functools
import numpy as np
import os
import sys

import dataloaders.new.data.load_ops as load_ops
from   dataloaders.new.data.load_ops import mask_if_channel_le
from   dataloaders.new.data.task_data_loading import load_and_specify_preprocessors

def get_inp_cfg( nopause=False ):
    cfg = {}

    # inputs
    cfg['input_dim'] = (256, 256)  # (1024, 1024)
    cfg['input_num_channels'] = 3
    cfg['input_domain_name'] = 'rgb'
    cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image
    cfg['input_preprocessing_fn_kwargs'] = {
        'new_dims': cfg['input_dim'],
        'new_scale': [-1, 1]
    }

    return cfg

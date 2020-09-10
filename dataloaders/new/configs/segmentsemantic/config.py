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
    cfg['only_target_discriminative'] = True
    cfg['target_domain_name'] = 'segmentsemantic'
    cfg['return_accuracy'] = True
    cfg['target_from_filenames'] = load_ops.semantic_segment_rebalanced

    # outputs
    cfg['target_num_channels'] = 17
    cfg['target_dim'] = (256, 256)  # (1024, 1024)
    cfg['target_from_filenames_kwargs'] = {
        'new_dims': (256,256),
        'domain' : 'segmentsemantic'
    }
    cfg['mask_by_target_func'] = True

    # masks

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors

    return cfg

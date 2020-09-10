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
    cfg['is_discriminative'] = True

    cfg['single_filename_to_multiple']=True

    # outputs
    cfg['target_dim'] = 9 # (1024, 1024)
    cfg['target_from_filenames'] = load_ops.vanishing_point_well_defined

    # input pipeline
    cfg['preprocess_fn'] = load_and_specify_preprocessors

    return cfg

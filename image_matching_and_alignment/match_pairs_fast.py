# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:42:57 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:15:29 2021

@author: Admin
"""
import torch
import matplotlib.cm as cm
import collections

import sys
sys.path.append('../')

from superGlue_model.matching import Matching
from superGlue_model.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def match_pairs(input_0_path, input_compare_path, matching, device):

    timer = AverageTimer(newline=True)

    # Load the image pair.
    # read_image(path, device, resize, rotation, resize_float):
    image0, inp0, scales0 = read_image(input_0_path, device, [-1], 0, False)
    image1, inp1, scales1 = read_image(input_compare_path, device, [-1], 0, False)

    timer.update('load_image')

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    timer.update('matcher')

    import numpy as np
    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    color = cm.jet(mconf)
    
    return mconf, mkpts0, mkpts1, color


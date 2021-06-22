# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:26:47 2021

@author: Admin
"""
import cv2
import os
import torch
from tqdm import tqdm
import numpy as np

# SuperGlue:
# ============================================================================
from match_pairs_fast import match_pairs
from image_utils import build_image_file_list_sorted
import numpy as np

import sys

sys.path.append('../')
from superGlue_model.matching import Matching
from superGlue_model.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot_fast,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from load_superGlue_model import load_model
# ============================================================================

from model_class import model
from match_class import match, get_matches
from get_average_image import get_Avg_Image
# RANSAC:
# ----------------------------------------------------------------------------
from RANSAC_flow_functions import load_pretrained_model, set_RANSAC_param 
from RANSAC_flow_functions import fine_alignnment, coarse_alignment
from RANSAC_flow_functions import show_coarse_alignment, show_fine_alignment, show_no_alignment
from PIL import Image
from PIL import ImageOps
from ransac_load import ransac_load
from ransac_flow_alignment import ransac_flow_coarse
# ----------------------------------------------------------------------------

# HOMOGRAPHY:
# ----------------------------------------------------------------------------
from homography_obj_warping import homography_match_obj
# ----------------------------------------------------------------------------

import heapq


def homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model):

    target_image = cv2.imread(target_image_path)
    target_image_name = os.path.splitext(os.path.basename(target_image_path))[0]

    if not os.path.exists(save_dir + '/homography_warped_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/homography_warped_image/')
    
    mconf, mkpts_target, mkpts_sample, color = get_matches(target_image_path, sample_image_path, model)
    match_obj = match(mkpts_target, mkpts_sample, mconf, target_image_path, sample_image_path, color) 
    #best_match_dict = match_obj.get_dict()
        
    if not os.path.exists(save_dir +'/_av_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/_av_image/') 
    
    # RANSAC FLOW on top images identified through matching
    target_image = Image.open(target_image_path).convert('RGB')
    target_image  = ImageOps.exif_transpose(target_image)
    '''  
    match_obj = match(best_match_dict['mkpts_target'], 
                      best_match_dict['mkpts_sample'],
                      best_match_dict['mconf'], 
                      best_match_dict['target_image_path'], 
                      best_match_dict['sample_image_path'],
                      best_match_dict['color'])'''
        
    homography_warp = homography_match_obj(match_obj, save_dir + '/homography_warped_image/')
    av = get_Avg_Image(homography_warp, target_image)
    av.save(save_dir +'/_av_image/average_'+match_obj.get_sample_image_name()+'.png')

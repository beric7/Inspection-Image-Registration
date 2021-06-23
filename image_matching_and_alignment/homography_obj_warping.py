# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:37:41 2021

@author: Admin
"""
import cv2
import sys
import os
sys.path.append('../')
from superGlue_model.utils import (compute_pose_error, compute_epipolar_error,
                                   estimate_pose, make_matching_plot_fast,
                                   error_colormap, AverageTimer, pose_auc, read_image,
                                   rotate_intrinsics, rotate_pose_inplane,
                                   scale_intrinsics)

def homography_match_obj(match_obj, save_dir):
    
    mkpts0 = match_obj.get_target_keypoint()
    mkpts1 = match_obj.get_sample_keypoint()
    target_image = match_obj.get_target_image()
    sample_image = match_obj.get_sample_image()
    color = match_obj.get_color()
    
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(0, 0),
        'Matches: {}'.format(len(mkpts0))]
    out = make_matching_plot_fast(cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY), 
                                  cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY), 
                                  0, 0, mkpts0, mkpts1, color, text,
                                  path=None, show_keypoints=False)
    
    
    sample_image_path = match_obj.get_sample_image_path()
    
    homography_matrix, _ = cv2.findHomography(mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=1)
    homography_warp = cv2.warpPerspective(sample_image, homography_matrix, (sample_image.shape[1], sample_image.shape[0]))
    save_path = save_dir + match_obj.get_sample_image_name() +'_warped.jpg'
    cv2.imwrite(save_path, homography_warp)
    
    if not os.path.exists(save_dir + 'matches/'): # if it doesn't exist already
        os.makedirs(save_dir + 'matches/')
    
    cv2.imwrite(save_dir + 'matches/' + match_obj.get_sample_image_name() + '_keypoints.jpg', out)
    return homography_warp
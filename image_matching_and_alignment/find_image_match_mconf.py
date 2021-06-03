# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:26:47 2021

@author: Admin
"""
import cv2
import os
import torch
from tqdm import tqdm

# SuperGlue:
# ============================================================================
from match_pairs_fast import match_pairs
from image_utils import build_image_file_list_sorted
import numpy as np

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
# ============================================================================


# RANSAC:
# ----------------------------------------------------------------------------
from RANSAC_flow_functions import load_pretrained_model, set_RANSAC_param 
from RANSAC_flow_functions import fine_alignnment, coarse_alignment
from RANSAC_flow_functions import show_coarse_alignment, show_fine_alignment, show_no_alignment
from PIL import Image
from PIL import ImageOps


# ----------------------------------------------------------------------------


class model():
    def __init__(self):
        self.matching = None
        self.device = None
        self.exemplar_num = None
    def set_matching(self, matching):
        self.matching = matching
    def set_device(self, device):
        self.device = device
    def set_exemplar_num(self,exemplar_num):
        self.exemplar_num = exemplar_num
        
    def get_matching(self):
        return self.matching
    def get_device(self):
        return self.device
    def get_exemplar_number(self):
        return self.exemplar_num

def ransac_load(resume_path, kernel_size, nb_point): 
    
    nb_scale = 7
    coarse_iter = 10000
    coarse_tolerance = 0.01
    min_size = 1000
    imageNet = False # we can also use MOCO feature here
    scale_r = 1.2
    
    ransac_network = load_pretrained_model(resume_path, kernel_size, nb_point)
    coarse_model = set_RANSAC_param(nb_scale, coarse_iter, coarse_tolerance, min_size, imageNet, scale_r)
    return coarse_model, ransac_network

def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))

def homography(mkpts0, mkpts1, sample_image, save_dir):
    homography_matrix, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=1)
    homography_warp = cv2.warpPerspective(sample_image, homography_matrix, (sample_image.shape[1], sample_image.shape[0]))
    cv2.imwrite(save_dir+'/warped/'+'warped.png', homography_warp)
    cv2.imwrite(save_dir+'/warped/'+'warped.png', homography_warp)
    
def ransac_flow(network, coarse_model, target_image, sample_image, sample_image_name, save_av_dir, save_dir, target_n):
    sample_coarse, sample_coarse_im, flow_coarse, grid, featt = coarse_alignment(network, coarse_model, sample_image, target_image)
    target_image = target_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    sample_image = sample_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/resize_target/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/resize_target/')
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/warped_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/warped_image/')
        
    sample_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'x.png')   
    sample_coarse_im.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'warped.png')
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'u.png')
        
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/resize_target/target.png')
    sample_image_fine, sample_image_fine_im = fine_alignnment(network, sample_coarse, featt, grid, flow_coarse, coarse_model)
    
    av = get_Avg_Image(sample_coarse_im, target_image)
    av.save(save_av_dir + 'average.png')
    
    # show_fine_alignment(sample_coarse_im, target_image, sample_coarse_im)
        
    return sample_image_fine_im

def load_model():
    
        'choices={indoor, outdoor}, these are the superglue weights'
        superglue = 'outdoor'
        
        'maximum number of keypoints detected by Superpoint'
        max_keypoints = 1024
        
        'SuperPoint keypoint detector confidence threshold'
        keypoint_threshold = 0.005
        
        'SuperPoint Non Maximum Suppression (NMS) radius'
        nms_radius = 4
        
        'Number of Sinkhorn iterations performed by SuperGlue'
        sinkhorn_iterations = 100
        
        'SuperGlue match threshold'
        match_threshold = 0.5
    
        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        # print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        
        matching = Matching(config).eval().to(device)
        return matching, device    

def get_matches(input_0_path, input_n_path, model):    
    mconf, mkpts0, mkpts1, color = match_pairs(input_0_path, input_n_path, model.get_matching(), model.get_device())   
    return mconf, mkpts0, mkpts1

def get_top_keypoints(list_0, list_1):
    
    # list 0 = mconf
    # list 1 = points
    dictionary = {}
    zipped = zip(list_0, list_1)
    dictionary = dict(zipped)
    ordered_list = sorted(dictionary.items(), reverse=True)
    ordered_list_100 = ordered_list[0:100]
    top_100_keypoints = []
    for value in ordered_list_100:
        top_100_keypoints.append((value[1][0], value[1][1]))
        
    return top_100_keypoints

def find_starting_frame(model, keypts_target, keypts_sample, target_image_path, 
                        sample_image_list, best_image_number, mconf_best):
    
    for sample_image_num in range(0, len(sample_image_list)): 
        sample_image_path = sample_image_list[sample_image_num]
        mconf, mkpts_target, mkpts_sample = get_matches(target_image_path, sample_image_path, model)
        number_keypoints = len(mconf)
    
        if number_keypoints >= 1:
            
            print('path: ', sample_image_path)
            print('length: ', len(mconf))
            
            if mconf_best > len(mconf):
                mconf = mconf_best
                best_image_number = sample_image_num
        else:
            continue
    return best_image_number
        

def find_peak(model, keypts_target, keypts_sample, target_image_path, sample_image_list, 
              bool_count, past_count, sample_image_number, best_image_number, mconf_best):
    # we check that the mconf number is large enough for a good comparison and that we are at least 10 frames past the 
    # best matched frame to get a peak value. 
    
    # the number of images we go past the peak before cutting it off. 
    END_past_count = 51
    
    if bool_count and mconf_best != None:
        sample_image_path = sample_image_list[best_image_number]
        print('final path: ', best_image_number)
        sample_image = cv2.imread(sample_image_path)
        return sample_image, best_image_number, keypts_target, keypts_sample, mconf_best
    else:
        sample_image_path = sample_image_list[sample_image_number]
        mconf, mkpts_target, mkpts_sample = get_matches(target_image_path, sample_image_path, model)
        number_keypoints = len(mconf)
        
        if number_keypoints >= 0:
            
            print('path: ', sample_image_path)
            print('length: ', len(mconf))
            
            if len(mconf) > mconf_best:
                keypts_target = mkpts_target
                keypts_sample = mkpts_sample
                mconf_best = len(mconf)
                past_count = 1
                bool_count = False
                best_image_number = sample_image_number
                
                try:
                   length_test = sample_image_list[sample_image_number+1] 
                   sample_image_number += 1
                except:
                    bool_count = True
                    

            else:
                try: 
                    length_test = sample_image_list[sample_image_number+1]
                    sample_image_number += 1
                    past_count += 1
                    if past_count == END_past_count:
                        bool_count = True
                except:
                    bool_count = True

        else:
              try: 
                    length_test = sample_image_list[sample_image_number+1]
                    sample_image_number += 1
                    past_count += 1
                    if past_count == END_past_count:
                        bool_count = True                    
              except:
                    bool_count = True
        
        return find_peak(model, keypts_target, keypts_sample, target_image_path, sample_image_list, 
                         bool_count, past_count, sample_image_number, best_image_number, mconf_best)
        
def find_matching_image(target_image_dir, sample_image_dir, save_dir, model, 
                        exemplar_image_num, coarse_model, ransac_network):
    
    # This is the directory we are using to find the best match to a target image.
    # In other words, this is the directory of the most recent inspection.
    sample_image_list, sample_image_names = build_image_file_list_sorted(sample_image_dir)
    
    # This is the directory of the target images we are comparing to. In other words
    # this is a directory of the past inspection. 
    target_image_list, target_image_names = build_image_file_list_sorted(target_image_dir)
    
    
    number_keypoint_max = 0
    # keypts: matching points between both images.
    keypts_target = 0
    keypts_sample = 0
    
    # mconf: the matching confidence for each keypoint pair. 
    mconf_best = 0
    
    # target_image_number: the count of the target images
    target_image_number = 0
    
    # sample_image_number: the count of sample_images we have gone through
    sample_image_number = 0
    best_image_number = 0
    
    # kernel
    KERNEL_HALF = 10
    
    best_mconf = 0
    
    # set exemplar image list max size    
    model.set_exemplar_num(exemplar_image_num)
    exemplar_sample_image_list = []
    
    # For each of the target images in the target image directory we are opening
    # and reading the image and storing it as a cv2 image object. We are also 
    # storing the name of the target image. 
    for target_n in tqdm(range(0, len(target_image_list))):
        target_image_path = target_image_list[target_n]
        target_image = cv2.imread(target_image_path)
        target_image_name = target_image_names[target_n]
        
        # we are initializing the start of the sequence within the sample dataset 
        # or the most recent insepction. We go through 'X' number of images from
        # the sample dataset one time to establish a perfect starting position. 
        # The initial test of X number of images is defined as a parameter. 
        if target_n == 0:
            print('hello')
            
        # here we set the past-peak counter to 0, since we are starting new
        bool_count = False
        past_count = 0
        mconf_best = 0
        
        if target_n == 0:
            best_start_number = find_starting_frame(model, keypts_target, keypts_sample, 
                                                     target_image_path, sample_image_list, best_image_number, mconf_best)
        
            # sample_image_number: assigned to the best starting point for the first target image.
            best_image_number = best_start_number
        
        sample_image_number = best_image_number - 10
        
        if sample_image_number < 0:
            sample_image_number = 0
        # We are finding the best-matched pairs using the current target image, 
        # and folder of sample inspection images. 
        sample_image, best_image_number, keypts_target, keypts_sample, mconf_best = find_peak(model, 
                                                                                    keypts_target, keypts_sample, 
                                                                                    target_image_path, 
                                                                                    sample_image_list, 
                                                                                    bool_count,
                                                                                    past_count,
                                                                                    sample_image_number,
                                                                                    best_image_number,
                                                                                    mconf_best)

        
        homography(keypts_target, keypts_sample, sample_image, save_dir)
        # create a file called frame_{}
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/exemplar_sample_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/exemplar_sample_image/')
            
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/exemplar_warped_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/exemplar_warped_image/') 
            
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/exemplar_av_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/exemplar_av_image/') 
        
        # --------------------------------------------------------------------
        # At this point we have image pairs which have met the criteria for being the 
        # best candidates for the match. There are a certain number of them (10). 
        # This number can be altered in the parameters for the code at the load_model(). 
        
        # mkpts0, mkpts1 
        # --------------------------------------------------------------------

        av_list = [-1,0,1]
        # RANSAC FLOW on top images identified through matching
        target_image = Image.open(target_image_path).convert('RGB')
        target_image  = ImageOps.exif_transpose(target_image)
        target_image.save(save_dir + '/frame_{}'.format(target_n)+'/'+target_image_name)
        
        for value in av_list:
            
            try:
                sample_image_path = sample_image_list[best_image_number + value]
                sample_image_name = sample_image_names[best_image_number + value]
                sample_image = Image.open(sample_image_path).convert('RGB')
                sample_image = ImageOps.exif_transpose(sample_image)
                sample_image.save(save_dir + '/frame_{}'.format(target_n)+'/exemplar_sample_image/'+sample_image_name)
                
                # network, coarse_model, target_image, sample_image, best_image_number, save_dir):
                save_av_dir = save_dir + '/frame_{}'.format(target_n)+'/exemplar_av_image/'
                sample_image_fine = ransac_flow(ransac_network, coarse_model, target_image, sample_image, sample_image_name, save_av_dir, save_dir, target_n)

                sample_image_fine.save(save_dir + '/frame_{}'.format(target_n)+'/exemplar_warped_image/'+sample_image_name)     
            except:
                continue

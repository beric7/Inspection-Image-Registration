# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:26:47 2021

@author: Admin
"""
import cv2
import os
import torch
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
# SuperGlue:
# ============================================================================
from match_pairs_fast import match_pairs
from image_utils import build_image_file_list_sorted
import numpy as np

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot_fast,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
# ============================================================================

import heapq


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

class match():
    def __init__(self, target_keypoint, sample_keypoint, mconf, target_image_path, sample_image_path, color):
        self.target_keypoint = target_keypoint
        self.sample_keypoint = sample_keypoint
        self.mconf = mconf
        self.target_image_path = target_image_path
        self.sample_image_path = sample_image_path
        self.target_image = cv2.imread(self.target_image_path)
        self.sample_image = cv2.imread(self.sample_image_path)
        self.color = color

    def set_target_keypoint(self, target_keypoint):
        self.target_keypoint = target_keypoint
    def set_sample_keypoint(self, sample_keypoint):
        self.sample_keypoint = sample_keypoint
    def set_mconf(self, mconf):
        self.mconf = mconf
        
    def get_target_keypoint(self):
        return self.target_keypoint
    def get_sample_keypoint(self):
        return self.sample_keypoint
    def get_mconf(self):
        return self.mconf   
    def get_target_image_path(self):
        return self.target_image_path
    def get_sample_image_path(self):
        return self.sample_image_path
    def get_target_image(self):
        return self.target_image
    def get_sample_image(self):
        return self.sample_image
    def get_dict(self):
        dictionary = {'mkpts_target': self.target_keypoint, 
                      'mkpts_sample': self.sample_keypoint, 
                      'mconf': self.mconf, 
                      'number_keypoints': len(self.mconf),
                      'target_image_path': self.target_image_path,
                      'sample_image_path': self.sample_image_path,
                      'color': self.color}
        return dictionary
    def get_sample_image_name(self):
        image_path = self.get_sample_image_path()
        image_name = os.path.basename(image_path).split('.')[0]
        return image_name
    def get_color(self):
        return self.color

def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))

def homography(match_obj, save_dir):
    
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
    cv2.imshow('', homography_warp)
    cv2.waitKey(0)
    save_path = save_dir + match_obj.get_sample_image_name() +'_warped.png'
    cv2.imwrite(save_dir + 'target.png', target_image)
    cv2.imwrite(save_path, homography_warp)
    cv2.imwrite(save_dir + match_obj.get_sample_image_name() + '_keypoints.png', out)
    

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
        sinkhorn_iterations = 200
        
        'SuperGlue match threshold'
        match_threshold = 0.8
    
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
    return mconf, mkpts0, mkpts1, color

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
        mconf, mkpts_target, mkpts_sample, color = get_matches(target_image_path, sample_image_path, model)
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
              bool_count, past_count, sample_image_number, mconf_smallest, best_match_dict_list):
    # we check that the mconf number is large enough for a good comparison and that we are at least 10 frames past the 
    # best matched frame to get a peak value. 
    # the number of images we go past the peak before cutting it off. 
    END_past_count = 51
    
    if bool_count and len(best_match_dict_list) != 0:
        return best_match_dict_list
    else:
        sample_image_path = sample_image_list[sample_image_number]
        mconf, mkpts_target, mkpts_sample, color = get_matches(target_image_path, sample_image_path, model)
        
        temp_match = match(mkpts_target, mkpts_sample, mconf, target_image_path, sample_image_path, color)
        best_match_dict_list.append(temp_match.get_dict())
        number_keypoints = len(temp_match.get_mconf())
        
        if number_keypoints >= 20:
            
            print('path: ', temp_match.get_sample_image_path)
            print('keypoints: ', number_keypoints)
            
            if number_keypoints > mconf_smallest:
                best_match_dict_list = heapq.nlargest(3, best_match_dict_list, key=lambda s: s['number_keypoints'])
                past_count = 1
                bool_count = False
                mconf_smallest = heapq.nsmallest(1, best_match_dict_list, key=lambda s: s['number_keypoints'])
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
                         bool_count, past_count, sample_image_number, mconf_smallest, best_match_dict_list)
        
def find_matching_image(target_image_dir, sample_image_dir, save_dir, model, 
                        exemplar_image_num):
    
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
    mconf_smallest = 0
    
    # target_image_number: the count of the target images
    target_image_number = 0
    
    # sample_image_number: the count of sample_images we have gone through
    sample_image_number = 0
    best_image_number = 0
    
    # kernel
    KERNEL_HALF = 10
    
    best_mconf = 0
    
    best_match_dict_list = []
    
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
        mconf_smallest = 0
        
        if target_n == 0:
            best_start_number = find_starting_frame(model, keypts_target, keypts_sample, 
                                                     target_image_path, sample_image_list, best_image_number, mconf_smallest)
        
            # sample_image_number: assigned to the best starting point for the first target image.
            best_image_number = best_start_number
        
        sample_image_number = best_image_number - 10
        
        if sample_image_number < 0:
            sample_image_number = 0
        # We are finding the best-matched pairs using the current target image, 
        # and folder of sample inspection images. 
        best_match_dict_list = find_peak(model, keypts_target, keypts_sample, target_image_path, 
                                         sample_image_list, bool_count, past_count,
                                         sample_image_number, mconf_smallest, best_match_dict_list)
        
            # create a file called frame_{}
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/exemplar_sample_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/exemplar_sample_image/')
            
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/homography_warped_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/homography_warped_image/') 
            
        if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/exemplar_av_image/'): # if it doesn't exist already
            os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/exemplar_av_image/') 
            
        for match_dict in best_match_dict_list:
            match_obj = match(match_dict['mkpts_target'], 
                               match_dict['mkpts_sample'],
                               match_dict['mconf'], 
                               match_dict['target_image_path'], 
                               match_dict['sample_image_path'],
                               match_dict['color'])
            
            homography(match_obj, save_dir + '/frame_{}'.format(target_n)+'/homography_warped_image/')

        # --------------------------------------------------------------------
        # At this point we have image pairs which have met the criteria for being the 
        # best candidates for the match. There are a certain number of them (10). 
        # This number can be altered in the parameters for the code at the load_model(). 
        
        # mkpts0, mkpts1 
        # --------------------------------------------------------------------
        target_image = Image.open(target_image_path).convert('RGB')
        target_image  = ImageOps.exif_transpose(target_image)
        target_image.save(save_dir + '/frame_{}'.format(target_n)+'/'+target_image_name)

        '''
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
  
            except:
                continue
        '''

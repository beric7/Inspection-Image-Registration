# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:26:47 2021

@author: Admin
"""
import cv2
import os
import torch
from tqdm import tqdm
#import Keypoint


from match_pairs_fast import match_pairs
from image_utils import build_image_file_list
import numpy as np

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
class model():
    def __init__(self):
        self.matching = None
        self.device = None
    def set_matching(self, matching):
        self.matching = matching
    def set_device(self, device):
        self.device = device
        
    def get_matching(self):
        return self.matching
    def get_device(self):
        return self.device
        
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
        sinkhorn_iterations = 20
        
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
    mconf, mkpts0, mkpts1 = match_pairs(input_0_path, input_n_path, model.get_matching(), model.get_device())
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
        

def find_peak(model, number_keypoint_max, keypoints_0, keypoints_1, sample_image_path, target_image_list, 
              past_count, target_image_number, mconf_best):
    # we check that the mconf number is large enough for a good comparison and that we are at least 10 frames past the 
    # best matched frame to get a peak value. 
    target_image_path = target_image_list[target_image_number+past_count]
    mconf, mkpts0, mkpts1 = get_matches(sample_image_path, target_image_path, model)

    number_keypoints = len(mconf)
    target_image = cv2.imread(target_image_path)
    return target_image, 0, mkpts0, mkpts1, mconf


def find_matching_image(target_image_dir, sample_image_dir, model):
    target_image_list, target_image_names = build_image_file_list(target_image_dir)
    sample_image_list, sample_image_names = build_image_file_list(sample_image_dir)
    number_keypoint_max = 0
    mkpts0 = 0
    mkpts1 = 0
    mconf_best = 0
    target_image_number = 0
    
    
    for sample_i in tqdm(range(0, len(sample_image_list))):
        sample_image_path = sample_image_list[sample_i]
        sample_image = cv2.imread(sample_image_path)
        sample_image_name = sample_image_names[sample_i]
        past_count = 0
        target_image, target_image_number, mkpts0, mkpts1, mconf_best = find_peak(model, number_keypoint_max, 
                                                                                    mkpts0, mkpts1, 
                                                                                    sample_image_path, 
                                                                                    target_image_list, 
                                                                                    past_count, target_image_number,
                                                                                    mconf_best)
        #topkeypoints_0 = np.asarray((get_top_keypoints(mconf_best, mkpts0)))
        #topkeypoints_1 = np.asarray((get_top_keypoints(mconf_best, mkpts1)))
        target_image_name = target_image_names[target_image_number]
        
        # create a file called frame_{}
        if not os.path.exists('./target_test/frame_{}'.format(sample_i)+'/sample_image/'): # if it doesn't exist already
            os.makedirs('./target_test/frame_{}'.format(sample_i)+'/sample_image/') 
            
        if not os.path.exists('./target_test/frame_{}'.format(sample_i)+'/warped/'): # if it doesn't exist already
            os.makedirs('./target_test/frame_{}'.format(sample_i)+'/warped/') 
        
        # add reference image and target image to directory
        cv2.imwrite('./target_test/frame_{}'.format(sample_i)+'/sample_image/target.png', target_image)
        cv2.imwrite('./target_test/frame_{}'.format(sample_i)+'/sample_image/sample.png', sample_image)
        
        # --------------------------------------------------------------------
        # NAZMUS your homography code goes here 
        # mkpts0, mkpts1 
        # --------------------------------------------------------------------
        homography_matrix, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=1)
        homography_warp = cv2.warpPerspective(sample_image, homography_matrix, (sample_image.shape[1], sample_image.shape[0]))
        cv2.imwrite('./target_test/frame_{}'.format(sample_i)+'/warped/'+target_image_name+str('_warped_to_')+sample_image_name, homography_warp)
        cv2.imwrite('./target_test/frame_{}'.format(sample_i)+'/sample_image/target_warped_to_sample.png', homography_warp)
        
'''        
# compare directories
sample_image_dir = './right/' # the new inspection
compare_image_dir = './left/' # the old inspection

# create model object, so we don't continuously load the model weights and devices
matching, device = load_model()
model = model()
model.set_device(device)
model.set_matching(matching)

# find matches... we get a list out of range error, at around ~62/65 images
# but I will fix that later...
find_matching_image(sample_image_dir, compare_image_dir, model)
'''
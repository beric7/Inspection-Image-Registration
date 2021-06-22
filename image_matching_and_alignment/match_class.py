# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:35:00 2021

@author: Admin
"""
import cv2
import os
from match_pairs_fast import match_pairs

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
    

def get_matches(input_0_path, input_n_path, model):    
    mconf, mkpts0, mkpts1, color = match_pairs(input_0_path, input_n_path, model.get_matching(), model.get_device())   
    return mconf, mkpts0, mkpts1, color
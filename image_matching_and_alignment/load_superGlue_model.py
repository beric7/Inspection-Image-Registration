# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:30:40 2021

@author: Admin
"""
import torch
import sys
sys.path.append('../')
from superGlue_model.matching import Matching

def load_model():
    
        'choices={indoor, outdoor}, these are the superglue weights'
        superglue = 'indoor'
        
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
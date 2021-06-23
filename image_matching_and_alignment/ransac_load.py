# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:29:47 2021

@author: Admin
"""
from RANSAC_flow_functions import load_pretrained_model, set_RANSAC_param 
def ransac_load(resume_path, kernel_size, nb_point): 
    
    # if we had more GPU power I think we couldget better results by utilizing a larger
    # number of steps and greater scaling factors. 
    
    nb_scale = 7 # this indicates thenumber of steps between the scales
    coarse_iter = 10000
    coarse_tolerance = 0.01
    min_size = 1500
    imageNet = False # we can also use MOCO feature here
    scale_r = 2.0 # this indicates the different scales 
    
    ransac_network = load_pretrained_model(resume_path, kernel_size, nb_point)
    coarse_model = set_RANSAC_param(nb_scale, coarse_iter, coarse_tolerance, min_size, imageNet, scale_r)
    return coarse_model, ransac_network
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:29:47 2021

@author: Admin
"""
from RANSAC_flow_functions import load_pretrained_model, set_RANSAC_param 
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
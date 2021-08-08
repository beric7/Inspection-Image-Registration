# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:25:03 2021

@author: Admin
"""

import os
from two_image_fine_alignment import fine_alignment_two_img
from load_superGlue_model import load_model
from model_class import model
from ransac_load import ransac_load
from tqdm import tqdm

base_folder = 'C://Users/Admin/OneDrive - Virginia Tech/Desktop/data_preliminary_experiments/'
target_image_path = base_folder + 'cropped_all points large increase.png'
source_image_path = base_folder + 'cropped_left small increase.png'
save_dir = base_folder + 'results/small_increase/'

if not os.path.exists(save_dir): # if it doesn't exist already
    os.makedirs(save_dir)

# create model object, so we don't continuously load the model weights and devices
matching, device = load_model()
model = model()
model.set_device(device)
model.set_matching(matching)


# RANSAC inputs:
resume_path = 'D://inspection-image-registration/ransac_flow_master/RANSAC-Flow-master/ransac_model/pretrained/' + 'KITTI_TestFT.pth' ## model for visualization
kernel_size = 7
nb_point = 4
coarse_model, ransac_network = ransac_load(resume_path, kernel_size, nb_point)
'''
for image_folder in tqdm(os.listdir(data_folder)):
    
    for image in tqdm(os.listdir(data_folder + image_folder)):
        sample_image_path = data_folder + image_folder + '/' + image
        # homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model, coarse_model, ransac_network)
        coarse_alignment_two_img(target_image_path, sample_image_path, save_dir + image_folder +'/',  model, coarse_model, ransac_network)
'''
fine_alignment_two_img(target_image_path, source_image_path, save_dir +'/',  model, coarse_model, ransac_network)
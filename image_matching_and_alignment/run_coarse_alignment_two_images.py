# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:47:30 2021

@author: Admin
"""

import os
from two_image_coarse_alignment import coarse_alignment_two_img
from load_superGlue_model import load_model
from model_class import model
from ransac_load import ransac_load
from tqdm import tqdm
# Opens a image in RGB mode
base_folder = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/10 ft/homography_warped_image/'
target_image_path = base_folder + '/targets/4ft_target/cropped_4_normal.png'
data_folder = base_folder + '/pre-processed_data/'
save_dir = base_folder + '/outputs_4ft_target/'

if not os.path.exists(save_dir): # if it doesn't exist already
    os.makedirs(save_dir)

# create model object, so we don't continuously load the model weights and devices
matching, device = load_model()
model = model()
model.set_device(device)
model.set_matching(matching)


# RANSAC inputs:
resume_path = base_folder + 'KITTI_TestFT.pth' ## model for visualization
kernel_size = 7
nb_point = 4
coarse_model, ransac_network = ransac_load(resume_path, kernel_size, nb_point)

for image_folder in tqdm(os.listdir(data_folder)):
    
    for image in tqdm(os.listdir(data_folder + image_folder)):
        sample_image_path = data_folder + image_folder + '/' + image
        # homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model, coarse_model, ransac_network)
        coarse_alignment_two_img(target_image_path, sample_image_path, save_dir +'/'+ image_folder +'/',  model, coarse_model, ransac_network)
        
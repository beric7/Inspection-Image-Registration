# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:47:30 2021

@author: Admin
"""

import os
from two_image_homography_alignment import homography_alignment_two_img
from load_superGlue_model import load_model
from model_class import model
from tqdm import tqdm
# Opens a image in RGB mode
base_folder = './data/'
target_image_path = base_folder + 'sample/close_up.png'
data_folder = base_folder + 'sample/'
save_dir = base_folder + 'results/'


if not os.path.exists(save_dir): # if it doesn't exist already
    os.makedirs(save_dir)
    
# create model object, so we don't continuously load the model weights and devices
matching, device = load_model()
model = model()
model.set_device(device)
model.set_matching(matching)


for image_folder in tqdm(os.listdir(data_folder)):
    if os.path.isdir(data_folder+image_folder):
        for image in tqdm(os.listdir(data_folder + image_folder)):
            sample_image_path = data_folder + image_folder + '/' + image
            # homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model, coarse_model, ransac_network)
            homography_alignment_two_img(target_image_path, sample_image_path, save_dir + image_folder +'/', model)
    else:
        for image in tqdm(os.listdir(data_folder)):
            sample_image_path = data_folder + image
            # homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model, coarse_model, ransac_network)
            homography_alignment_two_img(target_image_path, sample_image_path, save_dir, model)
        
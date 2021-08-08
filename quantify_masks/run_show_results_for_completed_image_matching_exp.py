# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:48:39 2020

@author: Eric Bianchi
"""

import os 
from show_results_ohev import*
from tqdm import tqdm   
import torch

# Load the trained model, you could possibly change the device from cpu to gpu if 
# you have your gpu configured.
saved_model = 'F://PROJECTS/corrosion segmentation/deeplabV3_plus/DeepLabV3plus_corrosion/stored_weights/l1_loss/weights_27.pt'
model = torch.load(saved_model, map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

# base directory
base = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/'
id = 'every_1_frame/beam_2_frames_source_many_output_l2/'

source_target_image_dir = base + 'beam_2_targets_many_frames_cut/'
save_target_image_dir = base + 'beam_2_targets_many_frames_predictions_cut/'
source_frames_image_dir = base + id

count = 0
for image_name in tqdm(os.listdir(source_target_image_dir)):
    print(image_name)
    image_path = source_target_image_dir + image_name
    destination_mask = save_target_image_dir + 'frame_{}'.format(count) + '/predicted_target_mask/'
    destination_overlays = save_target_image_dir + 'frame_{}'.format(count) + '/combined_overlays/'
    destination_ohev = save_target_image_dir + 'frame_{}'.format(count) + '/ohev/'
    destination_fit_im = save_target_image_dir + 'frame_{}'.format(count) + '/fit_image/'
    generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev, destination_fit_im)
    count += 1

count = 0

# HOMOGRAPHY
for frame_name in tqdm(os.listdir(source_frames_image_dir)):
    frame_dir = source_frames_image_dir + frame_name + '/homography_warped_cut_im/'
    for image_name in os.listdir(frame_dir):
        image_path = frame_dir + image_name
        destination_mask = source_frames_image_dir + frame_name + '/homography_predicted_cut_sample_masks/'
        destination_overlays = source_frames_image_dir + frame_name + '/homography_combined_cut_overlays/'
        destination_ohev = source_frames_image_dir + frame_name + '/homography_cut_ohev/'
        destination_fit_im = source_frames_image_dir + frame_name + '/homography_cut_fit_pred_image/'
        generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev, destination_fit_im)   
        
'''
# COARSE
for frame_name in tqdm(os.listdir(source_frames_image_dir)):
    frame_dir = source_frames_image_dir + frame_name + '/coarse_warped_image/'
    for image_name in os.listdir(frame_dir):
        image_path = frame_dir + image_name
        destination_mask = source_frames_image_dir + frame_name + '/coarse_predicted_sample_masks/'
        destination_overlays = source_frames_image_dir + frame_name + '/coarse_combined_overlays/'
        destination_ohev = source_frames_image_dir + frame_name + '/coarse_ohev/'
        destination_fit_im = source_frames_image_dir + frame_name + '/coarse_fit_pred_image/'
        generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev, destination_fit_im)  
        

# FUSION
for frame_name in tqdm(os.listdir(source_frames_image_dir)):
    frame_dir = source_frames_image_dir + frame_name + '/fusion_warped_image/'
    for image_name in os.listdir(frame_dir):
        image_path = frame_dir + image_name
        destination_mask = source_frames_image_dir + frame_name + '/fusion_predicted_sample_masks/'
        destination_overlays = source_frames_image_dir + frame_name + '/fusion_combined_overlays/'
        destination_ohev = source_frames_image_dir + frame_name + '/fusion_ohev/'
        destination_fit_im = source_frames_image_dir + frame_name + '/fusion_fit_pred_image/'
        generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev, destination_fit_im)  '''
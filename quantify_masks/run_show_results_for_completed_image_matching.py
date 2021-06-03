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
model = torch.load(f'./stored_weights_plus/var_3plus/var_3plus_weights_30.pt', map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

source_target_image_dir = './targets/'
source_frames_image_dir = './results/'

count = 0
for image_name in tqdm(os.listdir(source_target_image_dir)):
    print(image_name)
    image_path = source_target_image_dir + image_name
    destination_mask = './results/frame_{}'.format(count)+'/predicted_target_mask/'
    destination_overlays = './results/frame_{}'.format(count)+'/combined_overlays/'
    destination_ohev = './results/frame_{}'.format(count)+'/ohev/'
    generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev)
    count += 1

count = 0
for frame_name in tqdm(os.listdir(source_frames_image_dir)):
    frame_dir = source_frames_image_dir + frame_name + '/exemplar_warped_image/'
    for image_name in os.listdir(frame_dir):
        image_path = frame_dir + image_name
        destination_mask = source_frames_image_dir + frame_name +'/predicted_sample_masks/'
        destination_overlays = source_frames_image_dir + frame_name +'/combined_overlays/'
        destination_ohev = source_frames_image_dir + frame_name +'/ohev/'
        generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev)   
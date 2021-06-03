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
model = torch.load('F://PROJECTS/crack segmentation/DeepLabV3plus_crack/saved_stored_weights/LCW_weights_15.pt', map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

source_frames_image_dir = './data/'

for image_name in tqdm(os.listdir(source_frames_image_dir)):
    image_path = source_frames_image_dir + image_name
    destination_mask = source_frames_image_dir +'/predicted_sample_masks/'
    destination_overlays = source_frames_image_dir +'/combined_overlays/'
    destination_ohev = source_frames_image_dir +'/ohev/'
    generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev)   
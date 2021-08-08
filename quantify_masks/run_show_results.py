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

source_frames_image_dir = '../result_test/'

for image_name in tqdm(os.listdir(source_frames_image_dir)):
    image_path = source_frames_image_dir + image_name
    destination_mask = './result_test/experiment/predicted_sample_masks/'
    destination_overlays = './result_test/experiment/combined_overlays/'
    destination_ohev = './result_test/experiment/ohev/'
    destination_fit_im = './result_test/experiment/fit_image/'
    generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev, destination_fit_im)   
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:06:39 2021

@author: Admin
"""
from PIL import Image
import os
# Opens a image in RGB mode

def ResizeMaxSize(I, minSize) : 

    w, h = I.size
    ratio = max(w / float(minSize), h / float(minSize)) 
    new_w, new_h = int(round(w/ ratio)), int(round(h / ratio)) 
    
    ratioW, ratioH = new_w / float(w), new_h / float(h)
    Iresize = I.resize((new_w, new_h), resample=Image.LANCZOS)
    
    return Iresize


# base directory
base = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/'
id = 'every_1_frame/beam_2/beam_2-1_frames_source_output_l1/'

source_target_image_dir = base + 'beam_2_targets_average/complete/'
save_target_image_dir = base + 'beam_2_targets_average/complete_split/'
source_frames_image_dir = base + id


for image_folder in os.listdir(source_target_image_dir):
    
    if not os.path.exists(save_target_image_dir + image_folder): # if it doesn't exist already
        os.makedirs(save_target_image_dir + image_folder) 
        
    for image in os.listdir(source_target_image_dir + image_folder):
        im = Image.open(source_target_image_dir + image_folder + '/'+ image)
        width, height = im.size
        
        # Setting the points for cropped image
        left = 0
        top = 0
        right = width / 2
        bottom = height
        
        im_left = im.crop((left, top, right, bottom))
        im_right = im.crop((right, top, width, bottom))
        
        im_left.save(save_target_image_dir + image_folder + '/left_' + image)
        im_right.save(save_target_image_dir + image_folder + '/right_' + image)
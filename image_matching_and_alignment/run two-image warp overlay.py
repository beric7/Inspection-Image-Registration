# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:13:01 2021

@author: Admin
"""
from PIL import Image
import numpy as np
import os

def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))

number = '2'
base_folder = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/outputs_4ft_target/8 ft/'
sample_path = base_folder + '/fusion/coarse_warped_image/'+'cropped_8ft_L_roll_warped.jpg'
target_path = base_folder + '/homography_warped_image_8 ft/'+'cropped_8ft_L_roll_warped.jpg'

sample_coarse_image = Image.open(sample_path)
target_image = Image.open(target_path)

av = get_Avg_Image(sample_coarse_image, target_image)

if not os.path.exists('./data/overlay/'): # if it doesn't exist already
    os.makedirs('./data/overlay/')
av.save('./data/overlay/' + 'average_{}'.format(number)+'.png')
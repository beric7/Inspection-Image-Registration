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
sample_path = './data/'+'image_1_mask_cracked.png'
target_path = './data/'+'image_1_cracked.jpeg'

sample_coarse_image = Image.open(sample_path)
target_image = Image.open(target_path)

av = get_Avg_Image(sample_coarse_image, target_image)

if not os.path.exists('./data/overlay/'): # if it doesn't exist already
    os.makedirs('./data/overlay/')
av.save('./data/overlay/' + 'average_{}'.format(number)+'.png')
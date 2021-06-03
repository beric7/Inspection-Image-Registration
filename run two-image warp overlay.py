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

number = '000569'
sample_coarse_path = './DATA/SYMPHONY DATASET/WARP 1/'+str(number)+'_warp_1.jpg'
target_path = './DATA/SYMPHONY DATASET/WARP 2/'+str(number)+'_warp_2.jpg'

sample_coarse_image = Image.open(sample_coarse_path)
target_image = Image.open(target_path)

av = get_Avg_Image(sample_coarse_image, target_image)

if not os.path.exists('./DATA/SYMPHONY DATASET/symphony_dataset_comparison/'): # if it doesn't exist already
    os.makedirs('./DATA/SYMPHONY DATASET/symphony_dataset_comparison/')
av.save('./DATA/SYMPHONY DATASET/symphony_dataset_comparison/' + 'average_{}'.format(number)+'.png')
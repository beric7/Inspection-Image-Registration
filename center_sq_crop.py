# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys
import os
from tqdm import tqdm 
import cv2

# rescale(source_image_folder, destination, dimension):
height = 844
width = 1500
source = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/targets/6ft_target/'
destination = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/'

if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
for filename in tqdm(os.listdir(source)):
    im1 = cv2.imread(source + '/' + filename) 
    
    x_mid = im1.shape[1] / 2
    x = height/2
    
    crop_img = im1[int(0):int(height), int(x_mid-x):int(x+x_mid)] 
           
    cv2.imwrite(destination + '/' + filename, crop_img)
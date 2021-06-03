# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:06:39 2021

@author: Admin
"""
from PIL import Image
import os
# Opens a image in RGB mode
folder = './ransac alignment test building/'
for image_folder in os.listdir(folder):
    
    for image in os.listdir(folder + image_folder):
        im = Image.open(folder + image_folder + '/'+ image)
        width, height = im.size
        
        # Setting the points for cropped image
        left = 0
        top = 0
        right = width / 2
        bottom = height
        
        im1 = im.crop((left, top, right, bottom))
        
        im1.save(folder + image_folder + '/cropped_' + image)
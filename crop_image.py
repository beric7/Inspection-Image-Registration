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



folder = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/original_data/'
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
        
        im1 = ResizeMaxSize(im1, 1500)
        
        im1.save(folder + image_folder + '/cropped_' + image)
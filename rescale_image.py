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
height = 560
width = 992
source = './target/'
destination = './'

if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
for filename in tqdm(os.listdir(source)):
    im1 = cv2.imread(source + '/' + filename) 
    
    image = cv2.resize(im1, (width,height))            
           
    cv2.imwrite(destination + '/' + filename, image)
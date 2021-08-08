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
source = 'left000020.png'
replace = './img_1.jpeg'
destination = './result_test/'

if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        

im1 = cv2.imread('./'+source) 
im2 = cv2.imread(replace) 

x_mid = im1.shape[1] / 2
x = im2.shape[1]/2

y_mid = im1.shape[0] / 2
y = im2.shape[0]/2

print (int(y_mid-y))
print (int(y+y_mid))

im1[int(y_mid-y):int(y_mid+y), int(x_mid - x):int(x_mid + x)] = im2
       
cv2.imwrite(destination + '/' + source, im1)
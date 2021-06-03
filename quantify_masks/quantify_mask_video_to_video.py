# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:07:24 2021

@author: Admin
"""

import cv2
import numpy as np
import os
import math
from PIL import Image

def resize_image(image):
    h = image.shape[0]
    w = image.shape[1]
    max_dim = max(h,w)
    
    if (max_dim) > 300:
        reduce = max_dim/300
        h = int(h/reduce)
        w = int(w/reduce)
        image = cv2.resize(image, (w, h)) 
    return image 

def convert_HSV(hsv_min, hsv_max):
   
    # Min
    min_h = hsv_min[0]
    min_h = min_h/360*179
    min_s = hsv_min[1]
    min_s = min_s/100*255
    min_v = hsv_min[2]
    min_v = min_v/100*255
    
    # Max
    max_h = hsv_max[0]
    max_h = max_h/360*179
    max_s = hsv_max[1]
    max_s = max_s/100*255
    max_v = hsv_max[2]
    max_v = max_v/100*255
    
    MIN = np.array([min_h, min_s, min_v],np.uint8)
    MAX = np.array([max_h, max_s, max_v],np.uint8)
    
    return MIN, MAX

def mask_detection(image, image_path):
    original = image
    # lime = 0,255,0
    # H: 0-179, S: 0-255, V: 0-255
    # Lime 'H' = 120
    
    # Green_lower = 80, 25, 25
    # Green_upper = 150, 100, 100
    # Pink_lower = 310, 10, 50
    # Pink_upper = 330, 100, 100
    # Red_lower = 0, 50, 50
    # Red_upper = 10, 100, 100
    # Blue_lower = 220, 44, 96
    # Blue_upper = 240, 98, 50
    
    # Blue Detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([100,60,60])
    hsv_max = np.array([130,255,225])
    
    # create the mask using the colors
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # erode and then dialiate to remove noise
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dialated_mask = cv2.dilate(erosion,kernel,iterations = 1)

    # get the pixel area of the mask
    area = np.sum(mask)/255
    
    # make the result of the dialated mask on the image.
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # show the masks
    # cv2.imshow('frame',image)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',result)
    # cv2.waitKey(0)
    
    # save the resulting image with the original side by side.
    folder, file = os.path.split(image_path)
    file, extension = file.split('.')
    newIm = np.hstack((original, result))
    cv2.imwrite("./combined_result" + "." + extension, newIm)
    
    print('area in pixels: ' + str(area))
    return area

def quantify_mask(image_path):
    # assuming that we have masked the spalling to a known pixel color
    # assuming that we have a known laser offset.
    # assuming that we have a workable laser color HSV range
    
    # print('laser offset: ' + str(laser_offset))
    # path1 = 'D://' + 'Pink_spalling.png'
    image = cv2.imread(image_path)
    resized_image = image 
    # cv2.imshow('blurred',resized_image)
    # cv2.waitKey(0)
    area_in_pixels = mask_detection(resized_image, image_path)
    
    print('The area size is approximately: ' + str(area_in_pixels))
    
    return area_in_pixels




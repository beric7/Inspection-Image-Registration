# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:53:12 2021

@author: Admin
"""

import numpy as np
from tqdm import tqdm
import cv2
import os
import math

# rescale(source_image_folder, destination, dimension):
height = 844
width = 1500
source = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/targets/8ft_sq_target/'
destination = 'C://Users/Admin/Documents/data_image_registration/6-18-2020-lab_test/'
# Create a larger black colored canvas

for filename in tqdm(os.listdir(source)):
    im1 = cv2.imread(source + '/' + filename) 
    pad_width = math.ceil((width - height) / 2)
    pad_width = [(0, 0), (pad_width, pad_width), (0,0)]
    img_padded = np.pad(im1, pad_width, 'constant')
    cv2.imwrite(destination + '/' + filename, img_padded)
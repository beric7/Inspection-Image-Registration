# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:40:28 2021

@author: Admin
"""
from PIL import Image
import numpy as np

def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))
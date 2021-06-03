# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:29:45 2021

@author: Admin
"""
from RANSAC_flow_functions import load_pretrained_model, set_RANSAC_param 
from RANSAC_flow_functions import fine_alignnment, coarse_alignment
from RANSAC_flow_functions import show_coarse_alignment, show_fine_alignment, show_no_alignment
from PIL import Image
from PIL import ImageOps

nbScale = 7
coarseIter = 10000
coarsetolerance = 0.01
minSize = 1000
imageNet = False # we can also use MOCO feature here
scaleR = 1.2 

# Without alignment
I1 = Image.open('./img/left000080.png').convert('RGB')
I1 = ImageOps.exif_transpose(I1)
I2 = Image.open('./img/left000090.png').convert('RGB')
I2  = ImageOps.exif_transpose(I2)

resumePth = './model/pretrained/KITTI_TestFT.pth' ## model for visualization
kernelSize = 7
nbPoint = 4

network = load_pretrained_model(resumePth, kernelSize, nbPoint)
coarseModel = set_RANSAC_param(nbScale, coarseIter, coarsetolerance, minSize, imageNet, scaleR)

show_no_alignment(I1, I2)

I1_coarse, I1_coarse_pil, flowCoarse, grid, featt = coarse_alignment(network, coarseModel, I1, I2)
show_coarse_alignment(I1_coarse_pil, I2, coarseModel)

I1_fine, I1_fine_pil = fine_alignnment(network, I1_coarse, featt, grid, flowCoarse, coarseModel)
show_fine_alignment(I1_fine_pil, I2, coarseModel)
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:59:34 2021

@author: Admin
"""
from find_image_match_homograpy import*
from find_image_match_homograpy import ransac_load



# Input information:
# ----------------------------------------------------------------------------
d = ''
target_image_dir = './data/'+d+'/target/' # the older inspection
sample_image_dir = './data/'+d+'/sample/' # the newer inspection
save_dir = './data/'+d+'/test/'
exemplar_image_num = 3
# ----------------------------------------------------------------------------

# create model object, so we don't continuously load the model weights and devices
matching, device = load_superglue_model()
model = model()
model.set_device(device)
model.set_matching(matching)


# RANSAC inputs:
resume_path = 'D://inspection-image-registration/ransac_flow_master/RANSAC-Flow-master/ransac_model/pretrained/KITTI_TestFT.pth' ## model for visualization
kernel_size = 7
nb_point = 4
coarse_model, ransac_network = ransac_load(resume_path, kernel_size, nb_point)


# find the best matching image pairs for each target image
find_matching_image(target_image_dir, sample_image_dir, save_dir, model, exemplar_image_num, coarse_model, ransac_network)
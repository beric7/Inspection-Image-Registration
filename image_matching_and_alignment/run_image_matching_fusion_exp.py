# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:59:34 2021

@author: Admin
"""
from find_image_match_fusion import*
# from RANSAC_flow_functions import load_pretrained_model as ransac_load
# Input information:
# ----------------------------------------------------------------------------
target_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_2_targets_reversed_many_frames/' # the older inspection
sample_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_2-1_frames_source/' # the newer inspection
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/every_1_frame/beam_2_frames_source_many_output_l2/'
exemplar_image_num = 3

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
find_matching_image(target_image_dir, sample_image_dir, save_dir, model, 
                    exemplar_image_num, coarse_model, ransac_network)

##############################################################################
'''
target_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_1_targets/' # the older inspection
sample_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_1-2_frames_source/' # the newer inspection
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/every_1_frame/beam_1-2_frames_source_output_l1/'
exemplar_image_num = 3

# find the best matching image pairs for each target image
find_matching_image(target_image_dir, sample_image_dir, save_dir, model, 
                    exemplar_image_num, coarse_model, ransac_network)

##############################################################################

target_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_1_targets/' # the older inspection
sample_image_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/beam_1-3_frames_source/' # the newer inspection
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/every_1_frame/beam_1-3_frames_source_output_l1/'
exemplar_image_num = 3

# find the best matching image pairs for each target image
find_matching_image(target_image_dir, sample_image_dir, save_dir, model, 
                    exemplar_image_num, coarse_model, ransac_network)'''
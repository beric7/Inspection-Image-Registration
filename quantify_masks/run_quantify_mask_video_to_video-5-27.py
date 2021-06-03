# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:31:18 2021

@author: Admin
"""
import os
import pandas as pd
from quantify_mask_video_to_video import quantify_mask

target_image_dir ='./lab_test_target/'
sample_image_dir = './lab_test/'
inspection_name = 'test'
count = 0
sample_list = []
target_list = []
count_list = []
dict_output_error = {}
for target_image_path in os.listdir(target_image_dir):
    
    target_image_path = target_image_dir + '/' + target_image_path
    area_in_pixels_target = quantify_mask(target_image_path)
    count_list.append(count)
    target_list.append(area_in_pixels_target)
    sample_image_folder = sample_image_dir + '/frame_'+str(count)+'/exemplar_warped_image/'
    temp_sample_list = []
    
    for sample_image_path in os.listdir(sample_image_folder):
        area_in_pixels_sample = quantify_mask(sample_image_folder+sample_image_path)
        temp_sample_list.append(area_in_pixels_sample)
        error = abs((area_in_pixels_target-area_in_pixels_sample)/area_in_pixels_target)
        print('error: {}',error)
        
    sample_list.append(temp_sample_list)
    count += 1


dict_output_error['frame'] = count_list
dict_output_error['target area'] = target_list
for sample in range(0,len(sample_list)):
    column = [item[sample] for item in sample_list]
    dict_output_error['sample area '+str(sample)] = column

output_df = pd.DataFrame(dict_output_error)

output_df.to_csv('./matching_error'+'.csv')



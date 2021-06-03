# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:31:18 2021

@author: Admin
"""
import os
import pandas as pd
import cv2
from quantify_mask_video_to_video import quantify_mask

folder = './DATA/close elongation no context/'
target_image_path = folder + '/target/cropped_base close elongation.png'
sample_image_dir = folder + '/sample/'
inspection_name = 'close elongation no context'
count = 0
sample_list = []
target_list = []
count_list = []
dict_output_error = {}

area_in_pixels_target, image = quantify_mask(target_image_path)
cv2.imwrite(folder + 'target.jpg', image)
count_list.append(count)
target_list.append(area_in_pixels_target)
temp_sample_list = []

for sample_image_folder in os.listdir(sample_image_dir):
    for sample_image in os.listdir(sample_image_dir + sample_image_folder):
        sample_image_name = sample_image.split('.')[0]
        area_in_pixels_sample, image = quantify_mask(sample_image_dir+sample_image_folder + '/' + sample_image)
        temp_sample_list.append(area_in_pixels_sample)
        error = abs((area_in_pixels_target-area_in_pixels_sample)/area_in_pixels_target)
        print('error: {}',error)
        cv2.imwrite(folder + sample_image_name + '.jpg', image)
        sample_list.append(temp_sample_list)


dict_output_error[inspection_name] = count_list
dict_output_error['target area'] = target_list
for sample in range(0,len(sample_list)):
    column = [item[sample] for item in sample_list]
    dict_output_error['sample area '+str(sample)] = column

output_df = pd.DataFrame(dict_output_error)

output_df.to_csv('./matching_error'+'.csv')



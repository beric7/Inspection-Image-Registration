# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:52:06 2021

@author: Admin
"""
import torch
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
from metric_evaluation_two_masks_ohev import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score, evaluate_ohevs

# BASE DIRECTORIES
base = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/'

# TYPE OF IMAGE REGISTRATION USED
list_types = ['homography']


##############################################################################

def main(source_frames_image_dir, target_image_dir, save_results_dir, type):
    
    f1_av_list = []
    iou_av_list = []
    confm_av_list = []
    
    for frame in os.listdir(source_frames_image_dir):
        
        sum_ohev = np.zeros((1,512,512))
        count = 0
        for ohev in os.listdir(target_image_dir + frame + '/ohev/'):
            ohev_path = target_image_dir + frame + '/ohev/' + ohev
            np_ohev = np.load(ohev_path)
            sum_ohev += np_ohev
            count+=1
        av_ohev = np.rint(sum_ohev / count)
        
        np.save(target_image_dir + frame + '/ohev/average.npy', av_ohev)
        
        target_ohev_path = target_image_dir + frame + '/ohev/average.npy'
        
        y_input_av = 0
        y_input_sum = 0
        count = 0
        
        input_path_list = []
        
        for sample_ohev_path in os.listdir(source_frames_image_dir + frame + '/' + type + '_ohev/'):
            
            save_path = source_frames_image_dir + frame + '/confusion matrix/confm_' + type + '_' + sample_ohev_path.split('.')[0] + '.png'
            
            if not os.path.exists(source_frames_image_dir + frame + '/confusion matrix/'): # if it doesn't exist already
                os.makedirs(source_frames_image_dir + frame + '/confusion matrix/') 
            
            input_ohev_path = source_frames_image_dir + frame + '/' + type + '_ohev/' + sample_ohev_path
            
            confm_sum, iOU, f1, y_input, y_true = evaluate_ohevs(input_ohev_path, target_ohev_path)
            print(input_ohev_path)
            print('IOU: ', iOU)
            print('f1 score: ', f1)
            print('confm_sum: ', confm_sum)
            print('================================================================')
            
            input_path_list.append(input_ohev_path)
            
            y_input_sum += y_input
            count += 1
            
            # plot_confusion_matrix(confm_sum, save_path, target_names=['Background', 'Fair', 'Poor', 'Severe'], normalize=True, title='Confusion Matrix')
            
        # end for
        
        # average scores and round to nearest integer
        y_input_av = (y_input_sum / count)
        y_input_av_int = np.rint(y_input_av).astype('int')
        
        # get the average iOU, f1 scores, and confusion matrix
        iOU_av = jaccard_score(y_true, y_input_av_int, average='weighted')
        f1_av = f1_score(y_true, y_input_av_int, average='weighted')
        confm_av = confusion_matrix(y_true, y_input_av_int, labels=[0,1,2,3])
        
        # append to greater list for eventual csv file creation
        iou_av_list.append(iOU_av)
        f1_av_list.append(f1_av)
        confm_av_list.append(confm_av.ravel())
        
        # save path for the confusion matrix, and save confusion matrix figure
        save_av = source_frames_image_dir + frame + '/confusion matrix/confm_' + type + '_average.png'
        
        plot_confusion_matrix(confm_av, save_av, target_names=['Background', 'Fair', 'Poor', 'Severe'], normalize=True, title='Average Confusion Matrix')
        
        # HEAT MAPS
        # ---------------------------------------------------------------------
        # average inputs
        heatmap_im = y_input_av.reshape(512,512)
        plt.imshow(heatmap_im, cmap='jet', interpolation='nearest')
        plt.clim(0,3) # number of classes
        plt.colorbar()
        plt.savefig(source_frames_image_dir + frame + '/' + frame + '_' + type + '_average.png')
        plt.close()
        
        # ground truth
        heatmap_im_gt = y_true.reshape(512,512)
        plt.imshow(heatmap_im_gt, cmap='jet', interpolation='nearest')
        plt.clim(0,3) # number of classes
        plt.colorbar()
        plt.savefig(source_frames_image_dir + frame + '/'+ frame + '_' + type + '_gt.png')
        plt.close()
        # ---------------------------------------------------------------------
        # create excel writer object
    
    # SAVE 'iOU' TO (xlsx)       
    iOU_av_df = pd.DataFrame(iou_av_list)
    writer_iou = pd.ExcelWriter(save_results_dir + type + '_average iOU scores' + '.xlsx')
    iOU_av_df.to_excel(writer_iou)
    writer_iou.save()

    # SAVE 'F1 SCORES' TO (xlsx)
    f1_av_df = pd.DataFrame(f1_av_list)
    writer_f1 = pd.ExcelWriter(save_results_dir + type + '_average F1 scores' + '.xlsx')
    f1_av_df.to_excel(writer_f1)
    writer_f1.save()

    # SAVE 'CONFUSION MATRIX' TO (xlsx)
    confm_av_df = pd.DataFrame(confm_av_list)
    writer_conf = pd.ExcelWriter(save_results_dir + type + '_average confusion matrix' + '.xlsx')
    confm_av_df.to_excel(writer_conf)
    writer_conf.save()
    
    confm_av_df = pd.DataFrame(confm_av_list)
    confm_av_df.to_csv(save_results_dir + type + '_average confusion matrix' + '.csv')


def compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir):
    
    for type in list_types:
        main(source_frames_image_dir, target_image_dir, save_results_dir, type)

# TARGET IMAGES
target_image_dir = base + 'beam_2_targets_many_frames_predictions/'
##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_2_every_1/' # special save id
save_results_dir = base + 'results_l1_many_average_reg/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir) 

# ID DIRECTORY
id = 'every_1_frame/beam_2_frames_source_many_output_l2/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

'''
# TARGET IMAGES
target_image_dir = base + 'beam_2_targets_reversed_many_frames_predictions/'
##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_2_every_1/' # special save id
save_results_dir = base + 'results_l1_many_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir) 

# ID DIRECTORY
id = 'every_1_frame/beam_2_frames_source_many_output_l2/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)
'''

'''
# TARGET IMAGES
target_image_dir = base + 'beam_1_targets_average_www_predictions/'
##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-1_every_1/' # special save id
save_results_dir = base + 'results_www_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir) 

# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-1_frames_source_output_www/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-2_every_1/' # special save id
save_results_dir = base + 'results_www_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
    
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-2_frames_source_output_www/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-3_every_1/' # special save id
save_results_dir = base + 'results_www_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
    
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-3_frames_source_output_www/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)
'''
'''
# TARGET IMAGES
target_image_dir = base + 'beam_1_targets_average_l2_predictions/'
##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-1_every_1/' # special save id
save_results_dir = base + 'results_l2_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir) 

# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-1_frames_source_output_l2/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-2_every_1/' # special save id
save_results_dir = base + 'results_l2_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
    
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-2_frames_source_output_l2/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-3_every_1/' # special save id
save_results_dir = base + 'results_l2_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
    
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-3_frames_source_output_l2/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)
 

# TARGET IMAGES
target_image_dir = base + 'beam_1_targets_average_l1_predictions/'
##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-1_every_1/' # special save id
save_results_dir = base + 'results_l1_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)

# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-1_frames_source_output_l1/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-2_every_1/' # special save id
save_results_dir = base + 'results_l1_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
    
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-2_frames_source_output_l1/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)

##############################################################################
# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_1-3_every_1/' # special save id
save_results_dir = base + 'results_l1_average/' + special_id

if not os.path.exists(save_results_dir): # if it doesn't exist already
    os.makedirs(save_results_dir)
# ID DIRECTORY
id = 'every_1_frame/beam_1/beam_1-3_frames_source_output_l1/'

# SOURCE IMAGES
source_frames_image_dir = base + id

compare_list(list_types, source_frames_image_dir, target_image_dir, save_results_dir)'''
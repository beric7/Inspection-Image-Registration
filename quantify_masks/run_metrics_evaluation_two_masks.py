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
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
from metric_evaluation_two_masks import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score

frame_dir = './results/'
##############################################################################

        
for frame in os.listdir(frame_dir):
    target_im = os.listdir(frame_dir + frame+'/predicted_target_mask/')[0]
    target_mask_path = frame_dir + frame + '/predicted_target_mask/' + target_im
    y_input_av = 0
    y_input_sum = 0
    count = 0
    input_path_list = []
    input_f1_score = []
    for sample_mask_path in os.listdir(frame_dir + frame +'/predicted_sample_masks/'):
        save = sample_mask_path
        save = frame_dir + frame + 'confusion matrix/confm_' + save.split('.')[0] + '.png'
        
        if not os.path.exists(frame_dir + frame + 'confusion matrix/'): # if it doesn't exist already
            os.makedirs(frame_dir + frame + 'confusion matrix/') 
        
        input_mask_path = frame_dir + frame +'/predicted_sample_masks/' + sample_mask_path
        
        iOU, f1, confm_sum, y_input, y_true_ = iterate_data(input_mask_path, target_mask_path)
        print(input_mask_path)
        print('IOU: ', iOU)
        print('f1 score: ', f1)
        print('confm_sum: ', confm_sum)
        print('================================================================')
        input_path_list.append(input_mask_path)
        input_f1_score.append(f1)
        y_input_sum += y_input
        count += 1
        plot_confusion_matrix(confm_sum, save, 
                              target_names=['Background', 'Concrete', 'Steel', 'Metal Decking'], 
                              normalize=True, title='Confusion Matrix')
    
    # y = y_input_sum.reshape(1,y_input_sum.shape[0])
    y_input_av = (y_input_sum / count)
    y_input_av_int = np.rint(y_input_av).astype('int')
    
    confm_av = confusion_matrix(y_true_, y_input_av_int, labels=[0,1,2,3])
    save = frame_dir + frame + 'confusion matrix/confm_' + save.split('.')[0] + '.png'
    plot_confusion_matrix(confm_av, save, target_names=['Background', 'Concrete', 'Steel', 'Metal Decking'], normalize=True, title='Average Confusion Matrix')
    
    
    import matplotlib.pyplot as plt

    heatmap_im = y_input_av.reshape(512,512)
    plt.imshow(heatmap_im, cmap='jet', interpolation='nearest')
    plt.clim(0,3)
    plt.colorbar()
    plt.savefig(frame_dir + frame + '/' + frame+'_average.png')
    plt.show()
    
    heatmap_im_gt = y_true_.reshape(512,512)
    plt.imshow(heatmap_im_gt, cmap='jet', interpolation='nearest')
    plt.clim(0,3)
    plt.colorbar()
    plt.savefig(frame_dir + frame + '/'+ frame+'_gt.png')
    plt.show()

'''
dict_output_error['frame'] = count_list
dict_output_error['target area'] = target_list
for sample in range(0,len(sample_list)):
    column = [item[sample] for item in sample_list]
    dict_output_error['sample area '+str(sample)] = column

output_df = pd.DataFrame(dict_output_error)

output_df.to_csv('./matching_error'+'.csv')
'''


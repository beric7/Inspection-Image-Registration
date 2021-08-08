# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:12:55 2021

@author: Admin
"""

import pandas as pd
import numpy as np

# BASE DIRECTORIES
base = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/'

# WHERE WE ARE SAVING THE CSV FILES
special_id = 'beam_2_every_1/' # special save id
save_results_dir = base + 'results_l1_many_average_cut/' + special_id


# TYPE OF IMAGE REGISTRATION USED
list_types = ['homography']

for type in list_types:
    
    # SAVE 'CONFUSION MATRIX' TO (CSV)
    confm_av_df = pd.read_csv(save_results_dir + type + '_average confusion matrix' + '.csv')
    print(type)
    consolidated = pd.DataFrame()
    consolidated['sum t fair'] = confm_av_df['4'] + confm_av_df['5'] + confm_av_df['6'] + confm_av_df['7']
    consolidated['sum t poor'] = confm_av_df['8'] + confm_av_df['9'] + confm_av_df['10'] + confm_av_df['11']
    consolidated['sum t severe'] = confm_av_df['12'] + confm_av_df['13'] + confm_av_df['14'] + confm_av_df['15']
    
    consolidated['sum t+1 fair'] = confm_av_df['1'] + confm_av_df['5'] + confm_av_df['9'] + confm_av_df['13']
    consolidated['sum t+1 poor'] = confm_av_df['2'] + confm_av_df['6'] + confm_av_df['10'] + confm_av_df['14']
    consolidated['sum t+1 severe'] = confm_av_df['3'] + confm_av_df['7'] + confm_av_df['11'] + confm_av_df['15']
    
    consolidated['total t'] = consolidated['sum t fair'] + consolidated['sum t poor'] + consolidated['sum t severe']
    consolidated['total t+1'] = consolidated['sum t+1 fair'] + consolidated['sum t+1 poor'] + consolidated['sum t+1 severe']

    consolidated['Pixel error fair'] = (consolidated['sum t+1 fair'] - consolidated['sum t fair'])
    consolidated['Pixel error poor'] = (consolidated['sum t+1 poor'] - consolidated['sum t poor'])
    consolidated['Pixel error severe'] = (consolidated['sum t+1 severe'] - consolidated['sum t severe'])

    consolidated['Error fair'] = consolidated['Pixel error fair'] / consolidated['sum t fair']
    consolidated['Error poor'] = consolidated['Pixel error poor'] / consolidated['sum t poor']
    consolidated['Error severe'] = consolidated['Pixel error severe'] / consolidated['sum t severe']
    
    consolidated['Error fair'] = np.where(consolidated['Error fair'].isin(['inf']),100, consolidated['Error fair'])
    consolidated['Error poor'] = np.where(consolidated['Error poor'].isin(['inf']),100, consolidated['Error poor'])
    consolidated['Error severe'] = np.where(consolidated['Error severe'].isin(['inf']),100, consolidated['Error severe'])

    consolidated['Weight fair'] = (consolidated['sum t fair'] + consolidated['sum t+1 fair']) / (consolidated['total t'] + consolidated['total t+1'])
    consolidated['Weight poor'] = (consolidated['sum t poor'] + consolidated['sum t+1 poor']) / (consolidated['total t'] + consolidated['total t+1'])
    consolidated['Weight severe'] = (consolidated['sum t severe'] + consolidated['sum t+1 severe']) / (consolidated['total t'] + consolidated['total t+1'])
    
    consolidated['Error (Delta) fair'] = consolidated['Error fair'] * consolidated['Weight fair']
    consolidated['Error (Delta)  poor'] = consolidated['Error poor'] * consolidated['Weight poor']
    consolidated['Error (Delta)  severe'] = consolidated['Error severe'] * consolidated['Weight severe']
    
    # consolidated.to_csv(save_results_dir + type + '_consolidated.csv', index=False)
    
    # create excel writer object
    writer = pd.ExcelWriter(save_results_dir + type + '_consolidated.xlsx')
    # write dataframe to excel
    consolidated.to_excel(writer)
    # save the excel
    writer.save()
    print('DataFrame is written successfully to Excel File.')
    
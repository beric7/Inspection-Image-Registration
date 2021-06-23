# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:42:43 2021

@author: Admin
"""
from RANSAC_flow_functions import fine_alignnment, coarse_alignment
import os
from get_average_image import get_Avg_Image

def ransac_flow_frame_fine(network, coarse_model, target_image, sample_image, sample_image_name, save_av_dir, save_dir, target_n):
    
    sample_coarse, sample_coarse_im, flow_coarse, grid, featt = coarse_alignment(network, coarse_model, sample_image, target_image)
    target_image = target_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    sample_image = sample_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/resize_target/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/resize_target/')
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/warped_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/warped_image/')
        
    sample_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'x.png')   
    sample_coarse_im.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'warped.png')
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'u.png')
        
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/resize_target/target.png')
    sample_image_fine, sample_image_fine_im = fine_alignnment(network, sample_coarse, featt, grid, flow_coarse, coarse_model)
    
    av = get_Avg_Image(sample_image_fine_im, target_image)
    av.save(save_av_dir + sample_image_name + '_fine_average.png')
    
    # show_fine_alignment(sample_coarse_im, target_image, sample_coarse_im)
        
    return sample_image_fine_im

def ransac_flow_frame_coarse(network, coarse_model, target_image, sample_image, sample_image_name, save_av_dir, save_dir, target_n):
    
    sample_coarse, sample_coarse_im, flow_coarse, grid, featt = coarse_alignment(network, coarse_model, sample_image, target_image)
    target_image = target_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    sample_image = sample_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/resize_target/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/resize_target/')
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/frame_{}'.format(target_n)+'/warped_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/frame_{}'.format(target_n)+'/warped_image/')
        
    sample_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'x.png')   
    sample_coarse_im.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'warped.png')
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/warped_image/' + 'u.png')
        
    target_image.save(save_dir + '/frame_{}'.format(target_n)+'/resize_target/target.png')

    av = get_Avg_Image(sample_coarse_im, target_image)
    av.save(save_av_dir + sample_image_name + '_coarse_average.png')
        
    return sample_coarse_im


def ransac_flow_fine(network, coarse_model, target_image, sample_image, sample_image_name, save_dir):
    
    sample_coarse, sample_coarse_im, flow_coarse, grid, featt = coarse_alignment(network, coarse_model, sample_image, target_image)
    target_image = target_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    sample_image = sample_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/resize_target/'): # if it doesn't exist already
        os.makedirs(save_dir + +'/resize_target/')
    # create a file called frame_{}
    if not os.path.exists(save_dir + '/warped_image/'): # if it doesn't exist already
        os.makedirs(save_dir + '/warped_image/')
               
    target_image.save(save_dir + '/resize_target/target.png')
    sample_image_fine, sample_image_fine_im = fine_alignnment(network, sample_coarse, featt, grid, flow_coarse, coarse_model)
    
    av = get_Avg_Image(sample_image_fine_im, target_image)
    av.save(save_dir + sample_image_name + '_fine_average.png')
        
    return sample_image_fine_im


def ransac_flow_coarse(network, coarse_model, target_image, sample_image, sample_image_name, save_dir):
    
    sample_coarse, sample_coarse_im, flow_coarse, grid, featt = coarse_alignment(network, coarse_model, sample_image, target_image)
    sample_coarse_im =sample_coarse_im.resize((target_image.size[0], target_image.size[1]))
    #target_image = target_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))
    #sample_image = sample_image.resize((sample_coarse_im.size[0], sample_coarse_im.size[1]))

    iden = save_dir.split('/')[-2]
    
    if not os.path.exists(save_dir + '/coarse_av_image_'+iden+'/'): # if it doesn't exist already
        os.makedirs(save_dir + '/coarse_av_image_'+iden+'/')
    
    av = get_Avg_Image(sample_coarse_im, target_image)
    av.save(save_dir + '/coarse_av_image_'+iden+'/' + sample_image_name + '_coarse_average.png')
    
    return sample_coarse_im
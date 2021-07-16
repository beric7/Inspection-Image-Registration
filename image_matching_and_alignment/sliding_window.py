# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:18:49 2019

@author: Eric Bianchi
"""

import cv2
import numpy as np
import math
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
# superglue
from match_pairs_fast import match_pairs
from load_superGlue_model import load_model

def prepSuperGlue():
    from model_class import model
    matching, device = load_model()
    model = model()
    model.set_device(device)
    model.set_matching(matching)
    return model, device

def rsmeGrid(mkpts0, error, stepSize_x, stepSize_y, number_x, number_y):
    rsmeGrid = np.zeros((number_y, number_x))
    keypoint_grid = np.zeros((number_y, number_x))

    mkpts0_y = mkpts0[:, 1]
    mkpts0_x = mkpts0[:, 0]
    
    grid_mkpts_x = mkpts0_x / stepSize_x
    grid_mkpts_x = (grid_mkpts_x).astype(int).reshape(grid_mkpts_x.shape[0], 1)
    grid_mkpts_y = mkpts0_y / stepSize_y
    grid_mkpts_y = (grid_mkpts_y).astype(int).reshape(grid_mkpts_y.shape[0], 1)
    
    keypoint_grid_coord = np.append(grid_mkpts_y, grid_mkpts_x, 1)
    
    count = 0
    for kpt in keypoint_grid_coord:
        
        y = kpt[0]
        x = kpt[1]
        rsmeGrid[y][x] = rsmeGrid[y][x] + error[count]
        keypoint_grid[y][x] = keypoint_grid[y][x] + 1
        count+=1
    rsmeGrid_mean = rsmeGrid
    for y in range(number_y):
        for x in range(number_x):
            if keypoint_grid[y][x] != 0:
                rsmeGrid_mean[y][x] = rsmeGrid[y][x] / keypoint_grid[y][x]
            else:
                rsmeGrid_mean[y][x] = 100

    return rsmeGrid_mean

def rsmeWindows(target_win_kpts, warped_win_kpts):
    dist = np.linalg.norm(target_win_kpts - warped_win_kpts, axis=1)
    av_total_err = np.mean(dist)
    median = np.median(dist)
    print('The median: {}', median)
    return dist, av_total_err

def build_image_file_list(TEST_IMAGES_DIR):
    imageFilePaths = []
    image_names = []
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGES_DIR + "/" + imageFileName)
        if imageFileName.endswith(".JPG"):
            imageFilePaths.append(TEST_IMAGES_DIR + "/" + imageFileName)
        if imageFileName.endswith(".png"):
            im = Image.open(TEST_IMAGES_DIR + "/" + imageFileName)
            rgb_im = im.convert('RGB')
            (head, tail) = imageFileName.split(".")
            rgb_im.save(TEST_IMAGES_DIR + "/" + head + '.jpg')
            imageFilePaths.append(TEST_IMAGES_DIR + "/" + head + '.jpg')
        if imageFileName.endswith(".jpeg"):
            imageFilePaths.append(TEST_IMAGES_DIR + "/" + imageFileName)
        # end if
        image_names.append(imageFileName)
    
    return imageFilePaths, image_names

def get_matches(input_0_path, input_n_path, model):    
    mconf, mkpts0, mkpts1, color = match_pairs(input_0_path, input_n_path, model.get_matching(), model.get_device())   
    return mconf, mkpts0, mkpts1, color

'''
sliding window
'''
def generate_mask(keypoint_grid, target_im, stepSize_x, stepSize_y, white_mask, threshold):

    mask_grid = np.zeros((target_im.shape[0], target_im.shape[1], target_im.shape[2]))
    
    x_count = 0
    y_count = 0
    for x in range(0, target_im.shape[1], stepSize_x):
        y_count = 0 
        for y in range(0, target_im.shape[0], stepSize_y):
            temp_kpt_win = keypoint_grid[[y_count],[x_count]]
            if temp_kpt_win >= threshold:
                mask_grid[y:y + stepSize_y, x:x + stepSize_x:] = white_mask
            y_count = y_count + 1
        
        x_count = x_count + 1
                
    return mask_grid

def generate_mask_error(error_grid, target_im, stepSize_x, stepSize_y, white_mask, threshold):
    mask_grid = np.zeros((target_im.shape[0], target_im.shape[1], target_im.shape[2]))

    x_count = 0
    y_count = 0
    for x in range(0, target_im.shape[1], stepSize_x):
        y_count = 0
        for y in range(0, target_im.shape[0], stepSize_y):
            temp_kpt_win = error_grid[[y_count], [x_count]]
            if temp_kpt_win <= threshold:
                mask_grid[y:y + stepSize_y, x:x + stepSize_x:] = white_mask
            y_count = y_count + 1

        x_count = x_count + 1

    return mask_grid

def slidingWindow_error(image_grid, error_grid, number_x, number_y, stepSize_x, stepSize_y):
    
    mask_grid = image_grid
    
    for x in range(number_x):
        for y in range(number_y):
            fill = error_grid[x,y]
            mask_grid[x][y] = (stepSize_x, stepSize_y, fill)
                
    return mask_grid
   
def smallestDivisor(n):
   n = int(n)
   a=[]
   for i in range(2,n+1):
       if(n%i==0):
           a.append(i)
           a.sort()
           print("Smallest divisor is:",a[0])
           return int(a[0])
      
def optimalSize(step, n):
    if n >= 12 or step <= 200:
        return step, n
    else:
        divider = smallestDivisor(step)
        step = step / divider
        n = divider * n
        if step == 1:
            return divider, int(n/divider)
        return optimalSize(step, n)
 
def imageGrid(image, stepSize_x, stepSize_y):
    
    grid = []
    for x in range(0, image.shape[1], stepSize_x):
        column = []
        for y in range(0, image.shape[0], stepSize_y):
            window = image[y:y + stepSize_y, x:x + stepSize_x:]
            column.append(window)
        grid.append(column)
    return grid

'''
@param mkpts: ordered by the y coordinate
'''
def keypointGrid(mkpts0, stepSize_x, stepSize_y, number_x, number_y):
    keypoint_grid = np.zeros((number_y, number_x))
    
    mkpts0_y = mkpts0[:, 1]
    mkpts0_x = mkpts0[:, 0]
    
    grid_mkpts_x = mkpts0_x / stepSize_x
    grid_mkpts_x = (grid_mkpts_x).astype(int).reshape(grid_mkpts_x.shape[0], 1)
    grid_mkpts_y = mkpts0_y / stepSize_y
    grid_mkpts_y = (grid_mkpts_y).astype(int).reshape(grid_mkpts_y.shape[0], 1)
    
    keypoint_grid_coord = np.append(grid_mkpts_y, grid_mkpts_x, 1)
    
    for kpt in keypoint_grid_coord:
        y = kpt[0]
        x = kpt[1]
        keypoint_grid[y][x] = keypoint_grid[y][x] + 1
    
    return keypoint_grid

def gridImage(image):
    x_im = image.shape[1]
    y_im = image.shape[0]
    
    gcd = math.gcd(x_im, y_im)

    stepSize_x, number_x = optimalSize(x_im, 1)
    stepSize_x = int(stepSize_x)
    stepSize_y, number_y = optimalSize(y_im, 1)
    stepSize_y = int(stepSize_y)
    
    
    grid_image = image # for drawing a rectangle
    for x in range(0, image.shape[1], stepSize_x):
        for y in range(0, image.shape[0], stepSize_y):
            grid_image = cv2.rectangle(grid_image, (x, y), (x + stepSize_x, y + stepSize_y), (0, 0, 255), 1) # draw rectangle on image
    
    return grid_image, stepSize_x, stepSize_y, number_x, number_y    

def createMaskWindow(stepSize_x, stepSize_y, fill):
    
    img = np.zeros([stepSize_x,stepSize_y,3],dtype=np.uint8)
    img.fill(fill)
    
    return img

def heatmap(input, image_name, image, save_dir, threshold, toggle):

    '''
    heatmap_im = cv2.resize(input, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    plt.imshow(heatmap_im, cmap='jet_r', interpolation='nearest')
    plt.clim(0,threshold*1.5)
    plt.colorbar()
    plt.savefig('./'+image_name+'_average.png')
    plt.show()
    
    heatmap_im = cv2.imread('./'+image_name+'_average.png')
    '''
    if toggle:
        heatmap_im = cv2.resize(input, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        plt.imshow(image)
        plt.imshow(heatmap_im, alpha=0.4, cmap='jet_r', interpolation='nearest')
        plt.clim(0,threshold*1.5)
        plt.colorbar()
        plt.savefig(save_dir + image_name + '_overlay_average.png')
        plt.show()
        plt.close()
    else:
        heatmap_im = cv2.resize(input, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        plt.imshow(image)
        plt.imshow(heatmap_im, alpha=0.4, cmap='jet', interpolation='nearest')
        plt.clim(0,threshold*10)
        plt.colorbar()
        plt.savefig(save_dir + image_name + '_overlay_average.png')
        plt.show()
        plt.close()
    
    return heatmap_im

def createBinaryWindow(stepSize_x, stepSize_y, fill):
    
    img = np.zeros([stepSize_x,stepSize_y],dtype=np.uint8)
    img.fill(fill)
    
    return img
'''
#@param base_np_image: the base image (expects the image to be a numpy array)
@param acceptable_region_mask: the acceptable region to mask the warped image (nunmpy array)
'''
def maskRegion(base_np_image, acceptable_region_mask):
    print()
    # load images
    base_img  = Image.fromarray(base_np_image)
    mask_img = Image.fromarray(acceptable_region_mask)
    
    # convert images
    #img_org  = img_org.convert('RGB') # or 'RGBA'
    mask_img = mask_img.convert('L')    # grayscale
    
    # add alpha channel    
    base_img = base_img.putalpha(mask_img)
    
    # save as png which keeps alpha channel 
    base_img.save('output-pillow.png')
    
    return base_img
    
def drawRecOnImage(image, coordList, stepSize, color):
    

    tmp = image # for drawing a rectangle
    for i in range(0,len(coordList)):
        x = coordList[i][0]*stepSize
        y = coordList[i][1]*stepSize
        cv2.rectangle(tmp, (x, y), (x + stepSize, y + stepSize), color, 2) # draw rectangle on image
    return tmp

def homography(mkpts1, mkpts0, sample_image):
    homography_matrix, _ = cv2.findHomography(mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=3)
    homography_warp = cv2.warpPerspective(sample_image, homography_matrix,
                                          (sample_image.shape[1], sample_image.shape[0]))

    return homography_warp

def keypt_main(source_image_directory, target_image_path, save_dir, threshold, resize=False, resize_shape=(512,512)):

    # reads the target image into memory
    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)

    cv2.imwrite('./temp_target.png', target_im)
    # determine the step size based on parameters. Creates an image with the grid
    # overlayed to demonstrate the box sizes.
    grid_image, stepSize_x, stepSize_y, number_x, number_y = gridImage(target_im)

    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)
    # create the image grid array of windows to be referenced
    image_grid = imageGrid(target_im, stepSize_x, stepSize_y)

    # create little black or white windows based on the size of the windows.
    # this is for masking the regions which do or do not meet the desired threshold
    black_mask = createMaskWindow(stepSize_y, stepSize_x, 0)
    white_mask = createMaskWindow(stepSize_y, stepSize_x, 255)
    
    # load the super glue model one time.
    superglue_model, device = prepSuperGlue()
    
    for image in os.listdir(source_image_directory):
        
        # make the source image path
        source_image_path = source_image_directory + image
        source_image = cv2.imread(source_image_path)

        if resize:
            source_image = cv2.resize(source_image, resize_shape)
        cv2.imwrite('./temp_source.png', source_image)
        # get the mathed keypoints and coordinate locations. Use these keypoints 
        # from the target image to determine which regions within the target
        # image will be shown or masked. The are [y, x]
        mconf, mkpts0, mkpts1, color = get_matches('./temp_target.png', './temp_source.png', superglue_model)

        # Keypoint grid
        keypoint_grid = keypointGrid(mkpts0, stepSize_x, stepSize_y, number_x, number_y)
        
        image_name = image.split('.')[0]
        np.save(save_dir + image_name + '_kpt_grid.npy', keypoint_grid)
        img_save_name = save_dir + image_name + '_' + 'masked_low_keypoint_density_.png'
        # heatmap of keypoint density
        heatmap_im = heatmap(keypoint_grid, 'keypoints', target_im, save_dir, threshold, True)

        # generate masks over low-density keypoints
        mask = generate_mask(keypoint_grid, target_im, stepSize_x, stepSize_y, white_mask, threshold)
        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(target_im, mask)
        cv2.imwrite(img_save_name, masked_image)

def rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=False, resize_shape=(512,512)):
    # reads the target image into memory
    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)

    cv2.imwrite('./temp_target.png', target_im)
    # determine the step size based on parameters. Creates an image with the grid
    # overlayed to demonstrate the box sizes.
    grid_image, stepSize_x, stepSize_y, number_x, number_y = gridImage(target_im)

    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)
    # create little black or white windows based on the size of the windows.
    # this is for masking the regions which do or do not meet the desired threshold
    black_mask = createMaskWindow(stepSize_y, stepSize_x, 0)
    white_mask = createMaskWindow(stepSize_y, stepSize_x, 255)

    # load the super glue model one time.
    superglue_model, device = prepSuperGlue()

    for image in os.listdir(source_image_directory):
        # make the source image path
        source_image_path = source_image_directory + image
        source_image = cv2.imread(source_image_path)

        if resize:
            source_image = cv2.resize(source_image, resize_shape)

        cv2.imwrite('./temp_source.png', source_image)

        # RSME grid
        mconf, mkpts0, mkpts1, color = get_matches('./temp_target.png', './temp_source.png', superglue_model)
        ###

        image_name = image.split('.')[0]
        print('image name: ' + image_name)
        error, av_total_err = rsmeWindows(mkpts0, mkpts1)

        img_save_name = save_dir + image_name + '_' + str(av_total_err) + '_' + 'masked_high_error_density.png'

        rsme_grid = rsmeGrid(mkpts0, error, stepSize_x, stepSize_y, number_x, number_y)
        # rsme_grid_t = rsme_grid.transpose()

        # heatmap of keypoint density
        heatmap(rsme_grid, image_name + '_' + str(av_total_err), target_im, save_dir, error_threshold, False)

        # generate masks over high-density error
        mask = generate_mask_error(rsme_grid, target_im, stepSize_x, stepSize_y, white_mask, error_threshold)
        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(target_im, mask)
        cv2.imwrite(img_save_name, masked_image)

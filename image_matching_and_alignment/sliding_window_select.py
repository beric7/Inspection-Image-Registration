# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:18:49 2019

@author: Eric Bianchi
"""

import cv2
import numpy as np
import math
import tkinter
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd

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

def rsme_grid(mkpts0, error, stepSize_x, stepSize_y, number_x, number_y):
    rsmeGrid = np.zeros((number_y, number_x))
    error_and_kpt_count = []
    org_grid = []
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
                error_and_kpt_count.append([keypoint_grid[y][x], rsmeGrid_mean[y][x]])
            else:
                rsmeGrid_mean[y][x] = 100

    return rsmeGrid_mean, error_and_kpt_count, keypoint_grid

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
    blur = target_im
    blur = cv2.blur(blur, (7,7))
    grey = blur
    grey = cv2.cvtColor(grey,cv2.COLOR_RGB2GRAY)
    grey = cv2.cvtColor(grey,cv2.COLOR_GRAY2RGB)
    
    red = np.zeros((target_im.shape[0], target_im.shape[1], target_im.shape[2]))
    red[..., 2] = 255
    red = red.astype(np.uint8)

    blended = cv2.addWeighted(src1=grey, alpha=0.7, src2=red, beta=0.3, gamma=0)
    
    error_agg = 0
    x_count = 0
    threshold_count = 0
    for x in range(0, target_im.shape[1], stepSize_x):
        y_count = 0
        for y in range(0, target_im.shape[0], stepSize_y):
            temp_error_win = error_grid[[y_count], [x_count]]
            if temp_error_win <= threshold:
                mask_grid[y:y + stepSize_y, x:x + stepSize_x:] = white_mask
                blended[y:y + stepSize_y, x:x + stepSize_x:] = target_im[y:y + stepSize_y, x:x + stepSize_x:]
                threshold_count += 1
                error_agg = error_agg + temp_error_win
            y_count = y_count + 1

        x_count = x_count + 1
    try:
        error_agg = error_agg / threshold_count
    except:
        error_agg = 1000
            

    return mask_grid, threshold_count, error_agg, blended

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
      
def optimalSize(step, n, set_min_n=12, set_max_step=64):
    if n >= set_min_n or step <= set_max_step:
        return step, n
    else:
        divider = smallestDivisor(step)
        step = step / divider
        n = divider * n
        if step == 1:
            return divider, int(n/divider)
        return optimalSize(step, n, set_min_n, set_max_step)


def optimal_square(input_square, input_shape):
    closest_square = input_square
    bool = True
    if input_shape % closest_square >= (input_square / 2):
        direction = True # up
    else:
        direction = False # down
    while(bool):
        if direction:
            test = input_shape%closest_square
            closest_square = closest_square + 1
        else:
            closest_square = closest_square - 1

        if input_shape%closest_square==0:
            bool = False
    print('closest size: {}', closest_square)
    return closest_square

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

def gridImage(image, number_x):
    x_im = image.shape[1]
    y_im = image.shape[0]
    
    gcd = math.gcd(x_im, y_im)

    stepSize_x = x_im / number_x
    stepSize_x = int(stepSize_x)

    output_stepSize = optimal_square(stepSize_x, y_im)

    stepSize_y = output_stepSize
    stepSize_y = int(stepSize_y)
    number_y = int(y_im / stepSize_y)
    
    
    grid_image = image # for drawing a rectangle
    for x in range(0, image.shape[1], stepSize_x):
        for y in range(0, image.shape[0], stepSize_y):
            grid_image = cv2.rectangle(grid_image, (x, y), (x + stepSize_x, y + stepSize_y), (0, 0, 255), 1) # draw rectangle on image
    
    return grid_image, stepSize_x, stepSize_y, number_y

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
        plt.imshow(heatmap_im, alpha=0.3, cmap='jet_r', interpolation='nearest')
        plt.clim(0,threshold*5)
        plt.colorbar()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.gcf()
        fig.set_size_inches(1.2*image.shape[1]*px, 1.2*image.shape[0]*px)
        fig.savefig(save_dir + image_name + '_overlay_average.png')
        fig.show()
        plt.close()
    else:
        heatmap_im = cv2.resize(input, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        plt.imshow(image)
        plt.imshow(heatmap_im, alpha=0.3, cmap='jet', interpolation='nearest')
        plt.clim(0,threshold*5)
        plt.colorbar()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.gcf()
        fig.set_size_inches(1.2*image.shape[1]*px, 1.2*image.shape[0]*px)
        fig.savefig(save_dir + image_name + '_overlay_average.png')
        fig.show()
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

def overlay_text(image, keypoint_grid, number_x, number_y, stepSize_x, stepSize_y):

    # Blue color in BGR
    color = (0, 255, 0)    
    # fontScale
    fontScale = 0.5
      
    # Line thickness of 2 px
    thickness = 1
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for y in range(number_y):
        for x in range(number_x):
            x_coord = stepSize_x * x + int(stepSize_x / 2)
            y_coord = stepSize_y * y + int(stepSize_y / 2)
            text = keypoint_grid[y][x]
            org = (x_coord, y_coord)

            image = cv2.putText(image, str(text), org, font, 
                               fontScale, color, thickness, cv2.LINE_AA)
    return image


def keypt_main(source_image_directory, target_image_path, save_dir, threshold, resize=False, resize_shape=(512,512), number_x=12):

    # reads the target image into memory
    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)

    cv2.imwrite('./temp_target.png', target_im)
    # determine the step size based on parameters. Creates an image with the grid
    # overlayed to demonstrate the box sizes.
    grid_image, stepSize_x, stepSize_y, number_y = gridImage(target_im, number_x)

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

def rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=False, resize_shape=(512,512), number_x=12):
    
    df_list = []
    
    if not os.path.exists(save_dir): # if it doesn't exist already
        os.makedirs(save_dir)
    
    # reads the target image into memory
    target_im = cv2.imread(target_image_path)

    if resize:
        target_im = cv2.resize(target_im, resize_shape)

    cv2.imwrite('./temp_target.png', target_im)
    
    # determine the step size based on parameters. Creates an image with the grid
    # overlayed to demonstrate the box sizes.
    grid_image, stepSize_x, stepSize_y, number_y = gridImage(target_im, number_x)

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

        rsmeGrid_mean, error_and_kpt_count, keypoint_grid = rsme_grid(mkpts0, error, stepSize_x, stepSize_y, number_x, number_y)

        
        df = pd.DataFrame(error_and_kpt_count)
        df_list.append(df)
        plt.plot(df.iloc[:,0], df.iloc[:,1], 'ro')
        plt.savefig(save_dir + image_name + '_plot_error.png')
        plt.close()
        

        keypoint_overlay = overlay_text(target_im, keypoint_grid, number_x, number_y, stepSize_x, stepSize_y)
        
        img_save_name = save_dir + image_name + '_keypoint_overlay.png'
        cv2.imwrite(img_save_name, keypoint_overlay)

        target_im = cv2.imread(target_image_path)
        source_im = cv2.imread(target_image_path)
    
        if resize:
            target_im = cv2.resize(target_im, resize_shape)
            
        # generate masks over high-density error
        mask, threshold_count, error_agg, grey = generate_mask_error(rsmeGrid_mean, target_im, stepSize_x, stepSize_y, white_mask, error_threshold)
        
        # heatmap of keypoint density
        heatmap(rsmeGrid_mean, image_name + '_' + 'error_agg_' + str(error_agg), keypoint_overlay, save_dir, error_threshold, False)

        grey = grey.astype(np.uint8)
        mask = mask.astype(np.uint8)
        masked_image_target = cv2.bitwise_and(target_im, mask)
        warped_image_target = cv2.bitwise_and(source_image, mask)

        target_img_save_name_mask = save_dir + image_name + '_' + 'thres_ct_' + str(threshold_count) + '_' + 'masked_high_error_density.png'
        warped_img_save_name_mask = save_dir + image_name + '_' + 'thres_ct_' + str(threshold_count) + '_' + 'warped_masked_high_error_density.png'

        cv2.imwrite(target_img_save_name_mask, masked_image_target)
        cv2.imwrite(warped_img_save_name_mask, warped_image_target)
        
        grey_img_save_name = save_dir  + image_name + '_' + 'thres_ct_' + str(threshold_count) + '_' + 'pretty_masked_high_error_density.png'
        cv2.imwrite(grey_img_save_name, grey)
    df_com = pd.DataFrame()
    
    for df in df_list:
        plt.plot(df.iloc[:,0], df.iloc[:,1], 'ro')
        df_com = df_com.append(df, ignore_index=True)
    
    plt.savefig(save_dir + 'COMBINED' + '_plot_error.png')
    plt.close()
    
    writer = pd.ExcelWriter(save_dir + '_consolidated.xlsx')
    # write dataframe to excel
    df_com.to_excel(writer)
    # save the excel
    writer.save()


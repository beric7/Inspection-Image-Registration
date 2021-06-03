# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:15:08 2021

@author: Admin
"""

from coarseAlignFeatMatch import CoarseAlign
import sys

sys.path.append('./utils')
import outil

sys.path.append('./ransac_model')
import model as model

import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle 
import pandas as pd
import kornia.geometry.transform.homography_warper as HMW

from itertools import product
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt 

## composite image    
def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))

def set_RANSAC_param(nbScale, coarseIter, coarsetolerance, minSize, imageNet, scaleR):     
    coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)
    return coarseModel


def load_pretrained_model(resumePth, kernelSize, nbPoint):
    Transform = outil.Homography
        
    ## Loading model
    # Define Networks
    network = {'netFeatCoarse' : model.FeatureExtractor(), 
               'netCorr'       : model.CorrNeigh(kernelSize),
               'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
               'netMatch'      : model.NetMatchability(kernelSize),
               }
        
    
    for key in list(network.keys()) : 
        network[key].cuda()
        typeData = torch.cuda.FloatTensor
    
    # loading Network 
    param = torch.load(resumePth)
    msg = 'Loading pretrained model from {}'.format(resumePth)
    print (msg)
    
    for key in list(param.keys()) : 
        network[key].load_state_dict( param[key] ) 
        network[key].eval()
    return network

def fine_alignnment(network, I1_coarse, featt, grid, flowCoarse, coarseModel):
    featsSample = F.normalize(network['netFeatCoarse'](I1_coarse.cuda()))
    corr12 = network['netCorr'](featt, featsSample)
    flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H
    
    flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    flowUp = flowUp.permute(0, 2, 3, 1)
    
    flowUp = flowUp + grid
    
    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()
    
    I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
    I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())
    return I1_fine, I1_fine_pil
      
def coarse_alignment(network, coarseModel, I1, I2):

    coarseModel.setSource(I1)
    coarseModel.setTarget(I2)
    
    I2w, I2h = coarseModel.It.size
    featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
                
    #### -- grid     
    gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
    gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
    grid = torch.cat((gridX, gridY), dim=3).cuda() 
    # grid = HMW.HomographyWarper(I2h,  I2w)
    
    bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
    bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
    
    flowCoarse = HMW.warp_grid(grid,bestPara)
    I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
    I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())
    return I1_coarse, I1_coarse_pil, flowCoarse, grid, featt
    
def show_coarse_alignment(I1_coarse_pil, I2, coarseModel):
    
    I1_coarse_pil.save('./coarse_.png')
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Coarse)')
    plt.imshow(I1_coarse_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(I2)
    plt.subplot(1, 3, 3)
    plt.title('Overlapped Image')
    plt.imshow(I2)
    
    av = get_Avg_Image(I1_coarse_pil, coarseModel.It)
    av.save('./average.png')
    plt.imshow(get_Avg_Image(I1_coarse_pil, coarseModel.It))
    plt.show()
    
def show_fine_alignment(I1_fine_pil, I2, coarseModel):
    
    I1_fine_pil.save('./fine_tuned.png')
    '''
    plt.figure(figsize=(20, 10))    
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Fine Alignment)')
    plt.imshow(I1_fine_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(I2)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    '''
    av = get_Avg_Image(I1_fine_pil, coarseModel.It)
    av.save('./average.png')
    #plt.title('Overlapped Image')
    #plt.imshow(get_Avg_Image(I1_fine_pil, coarseModel.It))
    #plt.savefig('./figure.png')
    plt.show()    
    
def show_no_alignment(I1, I2):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(I1)
    plt.axis('off')
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(I2)
    plt.axis('off')
    plt.title('Target Image')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(get_Avg_Image(I1.resize(I2.size), I2))
    plt.title('Overlapped Image')
    plt.show()





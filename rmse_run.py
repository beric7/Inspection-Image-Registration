# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:38:12 2021

@author: Admin
"""

from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.measure import compare_mse as mse
import os
import pandas as pd

def rmsdiff(im1, im2):
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(im1), img_as_float(im2)))

target = '8ft'
target_2 = '8'
compare = '4 ft'
compare_2 = '4ft'
compare_3 = '4'

im2= Image.open('./targets/'+target+'_target/cropped_'+target_2+'_normal.png').convert('RGB')
coarse = []
homography = []
fusion = []
for image in os.listdir('./outputs_'+target+'_target/'+compare+'/coarse_warped_image_'+compare+'/'):
    im1 = Image.open('./outputs_'+target+'_target/'+compare+'/coarse_warped_image_'+compare+'/'+image)
    rsme = rmsdiff(im1, im2)
    coarse.append(rsme)
    print(image + ' Coarse Alignment: {}'.format(rsme))

for image in os.listdir('./outputs_'+target+'_target/'+compare+'/homography_warped_image_'+compare+'/'):
    im1 = Image.open('./outputs_'+target+'_target/'+compare+'/homography_warped_image_'+compare+'/'+image)
    rsme = rmsdiff(im1, im2)
    homography.append(rsme)
    print(image + ' Homography: {}'.format(rsme))

for image in os.listdir('./outputs_'+target+'_target/'+compare+'/fusion/coarse_warped_image_fusion/'):
    im1 = Image.open('./outputs_'+target+'_target/'+compare+'/fusion/coarse_warped_image_fusion/'+image)
    rsme = rmsdiff(im1, im2)
    fusion.append(rsme)
    print(image = ' Fusion: {}'.format(rsme))
    
df = pd.DataFrame(list(zip(coarse, homography, fusion)),columns =['Coarse', 'Homography', 'Fusion'])

df.to_csv('./RSME-scores-target{}-compare{}.csv'.format(target, compare_2))
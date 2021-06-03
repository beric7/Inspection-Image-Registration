# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:21:23 2021

@author: Admin
"""

import os
import shutil
count = 0
folder = './run_1/'
destination = './run_1_1Hz/'

for image in os.listdir(folder):
    if count%15 == 0:
        shutil.copyfile(folder + image, destination+image)
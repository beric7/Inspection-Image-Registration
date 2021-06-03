# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:46:14 2021

@author: Eric Bianchi
"""

from svo_export_ import main

# main(svo_input_path, output_path, export_mode=2)

svo_input_path = './trial_run.svo'

output_path = './output/trail_1_LEFT/'

main(svo_input_path, output_path, export_mode=2)
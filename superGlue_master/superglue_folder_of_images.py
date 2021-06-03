# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:05:17 2021

@author: Admin
"""

#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

# --------------------------------------------------------------------
# Eric Bianchi tested on his GTX3070 and got a average of 5.8 FPS for both videos as input streams with unlimited keypoints
# cutting out the showing screen didnot speed anything up
# got 7-8 FPS using only 50 keypoints
# 
# --------------------------------------------------------------------
torch.set_grad_enabled(False)
# superglue = {'indoor, outdoor'}
def superGlue(base_input, new_input, superglue, output_dir=None, description='SuperGlue test'):

    # Glob if a directory of images is specified
    image_glob =['*.png', '*.jpg', '*.jpeg']
    
    # SuperPoint Non Maximum Suppression (NMS) radius
    nms_radius = 4
    
    # Number of Sinkhorn iterations performed by SuperGlue
    sinkhorn_iterations = 20
    
    # threshold for matches
    match_threshold = 0.2
    
    # keypoint threshold
    keypoint_threshold = 0.005
    
    # max number of keypoints 
    max_keypoints = 10
    
    # Show the detected keypoints
    show_keypoints = False
    
    # Do not display images to screen. Useful if running remotely
    no_display = True
    # Resize the input image before running inference. If two numbers, 
    # resize to the exact dimensions, if one number, resize the max 
    # dimension, if -1, do not resize'
    resize = [512,512]
    
    # Maximum length if input is a movie or directory
    max_length = 1000000 
    
    # Images to skip if input is a movie or directory
    skip = 1 
    
    force_cpu = False

    print(no_display)
    print(force_cpu)
    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    #Defining a VideoStreamer class vs1 for "image1" (test image/video)
    vs1 = VideoStreamer(base_input, resize, skip,
                       image_glob, max_length)
    # Defining a VideoStreamer class vs0 for "image0" (reference image/video) "test.mp4" is the test file Skipping every 29 frames to run at 1 FPS
    vs0 = VideoStreamer(new_input, resize, skip,
                        image_glob, max_length)
    frame0, ret0 = vs0.next_frame()
    # assert ret0, 'Error when reading the first frame (V2) (try different --input?)'
    #Uploading data to CUDA/GPU
    frame_tensor0 = frame2tensor(frame0, device)
    last_data0 = matching.superpoint({'image': frame_tensor0})
    last_data0 = {k + '0': last_data0[k] for k in keys}
    last_data0['image0'] = frame_tensor0 
    last_frame0 = frame0
    last_image_id0 = 0

    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True: #Running the loop for video imagery
        #Getting the next test frame
        frame1, ret1 = vs1.next_frame()
        if not ret1:
            print('Finished demo_superglue.py')
            break
        #Getting the next reference frame
        '''
        frame0, ret0 = vs0.next_frame()
        if not ret0:
            print('Finished demo_superglue.py')
            break
        '''
        timer.update('data')
        stem1, stem0 = vs1.i - 1, vs0.i - 1
        #Uploading data to CUDA/GPU
        frame_tensor1 = frame2tensor(frame1, device)

        #Matching keypoints of the reference video to the test video
        pred = matching({**last_data0, 'image1': frame_tensor1})
        kpts0 = last_data0['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            frame0, frame1, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=show_keypoints, small_text=small_text)

        # this may be a place where we could run the analysis faster. 
        # if we set opt to no_display, I think that we can get it running quicker.
        if not no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs0.cleanup()
                vs1.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data0 = {k+'0': pred[k+'1'] for k in keys}
                last_data0['image0'] = frame_tensor1
                last_frame0 = frame1
                last_image_id0 = (vs1.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                show_keypoints = not show_keypoints

        timer.update('viz')
        timer.print()

        if output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs0.cleanup()
    vs1.cleanup()

# if possible to bridge structure
superGlue('./test_images/base/256x256/', './test_images/new/256x256/', 'indoor', './output_dir/')
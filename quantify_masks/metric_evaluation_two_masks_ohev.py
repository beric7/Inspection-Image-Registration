# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:14:27 2020

@author: Eric Bianchi
"""
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
import numpy as np
import os
import cv2
import itertools
from scipy.sparse import diags

def plot_confusion_matrix(cm,
                          save,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.clim(0, 1)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save)
    plt.close()
    # plt.show()

def process_im(image_path):
    
    image = cv2.imread(image_path)
    img = image.transpose(2,0,1)
    
    return img

def evaluate_ohevs(input_ohev_path, target_ohev_path):
    
    # ohev
    target_ohev = np.load(target_ohev_path)
    input_ohev = np.load(input_ohev_path)

    y_true = target_ohev.ravel()
    y_input = input_ohev.ravel()
    
    confm = confusion_matrix(y_true, y_input, labels=[0,1,2,3])
    iOU = jaccard_score(y_true, y_input, average='weighted')
    f1 = f1_score(y_true, y_input, average='weighted')
    
    return confm, iOU, f1, y_input, y_true


def make_image_list(image_dir):
    image_list_array = []
    for image_name in os.listdir(image_dir):
        image_list_array.append(image_name)
    return image_list_array

def iterate_data(input_ohev_path, target_ohev_path):

    confm, iOU, f1, y_input, y_true = evaluate_ohevs(input_ohev_path, target_ohev_path)
    
    return iOU, f1, confm, y_input, y_true

def spectrum_score_cut(confm_sum):
    array_1 = [0]
    array_2 = [0]
    for i in range(confm_sum.shape[0]):
        array_1.append(i+1)
        array_1.insert(0,i+1)
        array_2.append(i+1)
        array_2.insert(0,0-(i+1))
    diagonal = diags(array_1, array_2, shape=(confm_sum.shape[0],confm_sum.shape[0])).toarray()
    diagonal = np.delete(diagonal, 0, 0) 
    confm_sum = np.delete(confm_sum, 0, 0)
    total = sum(sum(confm_sum))
    norm_confm_sum = confm_sum/total
    
    spectrum_matrix = norm_confm_sum * diagonal
    spectrum_score = float(np.sum(spectrum_matrix))
    spectrum_score_norm = spectrum_score / i 
    print('spectrum score: {:0.4f}'.format(spectrum_score_norm))
    return spectrum_score_norm

def spectrum_score(confm_sum):
    array_1 = [0]
    array_2 = [0]
    for i in range(confm_sum.shape[0]):
        array_1.append(i+1)
        array_1.insert(0,i+1)
        array_2.append(i+1)
        array_2.insert(0,0-(i+1))
    diagonal = diags(array_1, array_2, shape=(confm_sum.shape[0],confm_sum.shape[0])).toarray()
    total = sum(sum(confm_sum))
    norm_confm_sum = confm_sum/total
    
    spectrum_matrix = norm_confm_sum * diagonal
    spectrum_score = float(np.sum(spectrum_matrix))
    spectrum_score_norm = spectrum_score / i 
    print('spectrum score: {:0.4f}'.format(spectrum_score_norm))
    return spectrum_score_norm



def spectrum_score_norm(confm_sum):
    array_1 = [0]
    array_2 = [0]
    for i in range(confm_sum.shape[0]):
        array_1.append(i+1)
        array_1.insert(0,i+1)
        array_2.append(i+1)
        array_2.insert(0,0-(i+1))
    diagonal = diags(array_1, array_2, shape=(confm_sum.shape[0],confm_sum.shape[0])).toarray()
    cm = confm_sum.astype('float') / confm_sum.sum(axis=1)[:, np.newaxis]
    
    spectrum_matrix = cm * diagonal
    spectrum_score = float(np.sum(spectrum_matrix))
    spectrum_score_norm = spectrum_score / i**2 
    print('spectrum score norm: {:0.4f}'.format(spectrum_score_norm))
    return spectrum_score_norm
    
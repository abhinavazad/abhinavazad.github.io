# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 03:07:16 2021

@author: AA086655
"""


import os
import random
import numpy as np
import cv2
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter 

from methods_model_training import aug_cxrs, makemydir, load_img_fromIDs, plot_model_hist, getImagesAndLabels, plot_test_maskout3
from Methods_img_proceess import prep_cxr_segmtn, morph, adap_equalize
import math


'''


This module Performs various Data Augmentaion for any given image.

It alters the following factors to augment any given image:
rotation_range, zoom_range, width_shift_range, shear_range,
height_shift_range, horizontal_flip, fill_mode.

For X-ray horizontal_flip = True
For OCR horizontal_flip = False

It saves the Augmented images by randomly altering some features at "des_path"

Parameters
----------
in_img_path : Path of the image to be Augmented.
des_path : Path where the augmented images are to be saved
total_image : Total number to augmented images to be generated.

Returns
-------
No returns, automatically saves the augmented images at "des_path" directory.
'''


n_times = 4

dim = 512
IMG_CHANNELS = 1


# Assigning Dataset paths
in_path = '/data/CXR/Training_split_2c/256adap_bal_aug/train/' 
in_path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 3c/train/"
out_path = in_path

dim = 512
def get_folders_only(path):
    return next(os.walk(path))[1]

#folder_name = os.path.basename(path)

classes = get_folders_only(in_path)
#classes = ['COVID', 'Normal', 'Viral Pneumonia']
#classes = ['COVID', 'Normal']
#classes = ['COVID']


#des_path = out_path + 'Aug_X' + str(n_times) + '_rand2/' + '/same_seed512'
#makemydir(des_path)

for n, class_ in tqdm(enumerate(classes), total=len(classes)):
    class_in_path = in_path + str(class_) + '/'
    ids = getImagesAndLabels(class_in_path)
    print(len(ids))
    try:
        lst.append(len(ids)) 
        print(lst)
    except:
        lst = [len(ids)]
        print(lst)
print('\nList of id lenghts for each class: ', lst, "\nMinimum lenght: ", min(lst),  "Maximum lenght: ", max(lst))

# Orignal*(Aug_times+1)
Aug_times = 4 # it will reflect exactly only for the class with the least images, Augmentation multiplicaiton times for other classes will be adaptive
# Saves the Augmented images in the same directory
for n, class_ in tqdm(enumerate(classes), total=len(classes)):
    class_in_path = in_path + str(class_) + '/'
    ids = getImagesAndLabels(class_in_path)

    n_times = math.ceil((min(lst)/lst[n])*(Aug_times+1)) -1
    
    print('\n', len(ids), " images in class ", class_, "Augmenting ", n_times, "times = ",n_times*lst[n] )
    
    if n_times!= 0:
        print('Augmenting one by one for each cxrs')
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            name = id_ + '.png'
            
            #aug_cxrs(id_, class_in_path, class_in_path, dim, n_times)
    else:
        print("No augmentation as the images in ", class_, " are  already near to or more than ",(Aug_times*min(lst)))
            




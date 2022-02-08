# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:08:17 2021

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

dim = 100
IMG_CHANNELS = 1


# Assigning Dataset paths
in_path ='C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/trail/prep/' 
out_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/trail/'
#out_path = in_path

ids = getImagesAndLabels(in_path)


des_path = out_path + 'Aug_X' + str(n_times) + '_rand4/' + '/rand_seed' + str(dim)
#makemydir(des_path)



# Use this snippet for Augmentation with repeated patterns
print('Grabing images and masks')
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    name = id_ + '.png'
    aug_cxrs(id_, in_path, des_path, dim, n_times)
    



# # 1. des_path out_path = in_path



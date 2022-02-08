# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 02:41:26 2021

@author: abhia
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


from methods_model_training import aug_cxr_n_mask, makemydir, load_img_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat, plot_test_maskout3



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

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1


cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'

# Assigning path of the Dataset to be augmented
path = 'D:/cerner/Datasets/Lungs Segementation training/all_raw_CXRs/RAW256/'

ids = getImagesAndLabels(path + cxrs)


des_path = path + 'Aug_X' + str(n_times) + '_rand/'
#makemydir(des_path + cxrs)
#makemydir(des_path + lung_masks )




'''
# Use this snippet for Augmentation with repeated patterns
print('Grabing images and masks')
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    name = id_ + '.png'
    aug_cxr_n_mask(id_, path, des_path, n_times)
    
'''


# Use this snippet(until the end of the code) for Augmentation with random patterns
total = len(ids)
# Get all the training images and masks assigned to x_train and y_test array
x = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
print('Grabing images and masks')
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    name = id_ + '.png'
    img = imread(path + cxrs + name)#[:,IMG_CHANNELS] 
    if img.shape != (IMG_HEIGHT, IMG_WIDTH):
        img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_AREA)
    x[n] = img  #Fill empty x_train with values from img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
    mask = imread(path + lung_masks + name)
    if mask.shape != (IMG_HEIGHT, IMG_WIDTH):
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)  
    y[n] = mask   
print('Grabing training images and masks : Done!')

# Preprocessing for image_data_generator.fit function
x = np.array(x)
x = np.expand_dims(x, axis = -1)
y = np.array(y)
y = np.expand_dims(y, axis = -1)


seed = 32



def blur(img):
    return (cv2.blur(img,(3,3)))

img_data_gen_args = dict(rotation_range=3,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode="nearest", brightness_range=[0.7,1.3])
                     #preprocessing_function= blur)

mask_data_gen_args = dict(rotation_range=3,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode="nearest", brightness_range=[0.7,1.3],
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 


image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(x, augment=True, seed=seed)


mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y, augment=True, seed=seed)

makemydir(des_path + '/prep_cxrs/')
makemydir(des_path + lung_masks )
         
image_generator = image_data_generator.flow(x, seed=seed, batch_size=n_times, save_to_dir=des_path + '/prep_cxrs/',save_prefix="cxr", save_format="png")
mask_generator = mask_data_generator.flow(y, seed=seed, batch_size=n_times, save_to_dir=des_path + '/LungsMasks/',save_prefix="cxr", save_format="png")



#os.chdir(des_path + '/prep_cxrs/')


# total output = batch_size*iter, by iter augmentation function moves to the next image
iter = len(ids)

i = 1
for e in image_generator:
    if (i == iter):
        break
    i = i +1
        


i = 1
for e in mask_generator:
    if (i == iter):
        break
    i = i +1




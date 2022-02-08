# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 01:50:07 2021

@author: AA086655
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:14:55 2021

@author: AA086655
"""

from keras.preprocessing.image import ImageDataGenerator
import keras
import cv2
from keras.models import Model


from keras.models import Sequential, load_model

import matplotlib.pyplot as plt
import numpy as np


import cv2
import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from varname import nameof
 
from tqdm import tqdm 
from skimage.transform import resize


from keras.models import model_from_yaml

from Methods_img_proceess import prep_cxr_segmtn, morph, adap_equalize
from methods_model_training import makemydir, load_img_fromIDs, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat_seg, plot_test_maskout3, plot_maskout3, plot_maskonly2


import errno
from datetime import datetime

from sklearn.model_selection import train_test_split
import os.path


def prep_cxr_classification(in_path, out_dirr, ids, seg_model,dim):
    """
    This is the all in one pre-processing fucntion for the calssification training
    Optional features: maskout = True if you only want predicted lungs region to be visible
                     : adaptive histogram equalization
                     

    Parameters
    ----------
    in_path : Dataset folder with all the Source images together in folders classwose
    out_dirr : Directory for the split datset
    ids : id names list of all the original images in in_path directory 
    seg_model : Lungs segmentation model
    dim : Output dimensions of the preprocessed imahes

    Returns
    -------
    None.

    """
    #x = np.zeros((len(ids), dim, dim), dtype=np.uint8)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        print(name)
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        image = cv2.imread(in_path + name, 0)#[:,IMG_CHANNELS]
        #print(in_path + name)
        size = image.shape
        img = image
        if img.shape != (256, 256): #Input shape required for Mask prediction seg_model
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
        img = adap_equalize(img)
        img = np.expand_dims(np.array(img), axis = 0)
        preds = seg_model.predict(img, verbose=1)
        preds_t = (preds > 0.5).astype(np.uint8)
        preds_t = np.squeeze(preds_t)
        post_process = False
        if post_process:
                preds_t = morph(morph, 1)
                preds_t = morph(morph, 2)
                preds_t = morph(morph, 1)
        #plt.imshow(preds_t)
        #plt.show()
        
        
        mask = resize(preds_t, size , mode='constant',  preserve_range=True)
        mask = (mask > 0).astype(np.uint8)
        #plt.imshow(mask)
        #plt.show()
        img_out, ratio = grabLungsbox(image, mask, maskout=False) # maskout= False for whole CXR of the lung region
        if ratio == 0:
            print("Cropping failed for: ", name)
        '''
        img_out1 = adap_equalize(img_out1)
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(img_out, cmap='gray')
        
        plt.subplot(232)
        plt.title('After pre-processing')
        plt.imshow(img_out1, cmap='gray')
        '''
        if img_out.shape != (dim, dim):
            interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
            img_out = cv2.resize(img_out, (dim, dim), interpolation = interpolation_type)
        
        img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
        #x[n] = img_out  #Fill empty x with values from img
#    return x



modelpath = os.getcwd() +'/b32-e150_X5703_06-19__21-53-43/'
modelname = "256_b32-e150_X5703_Raw_blur_prep_model_lungs_segmtn_06-19__21-53-43"

#modelpath = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Selected results/b8_X6079_06-17__08-54-16/"
#modelname = "256_b8_X6079_Raw_blur_prep_model_lungs_segmtn_06-17__08-54-16"

# load model
yaml_file = open(modelpath + modelname + '.yaml', 'r')
model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(model_yaml)
# load weights into new model
model.load_weights(modelpath + modelname + '.h5')
print("Loaded model from disk")


dim = 256

#in_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/trail/cxrs/'
#in_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19 Radiography Database trial/' #COVID-19 Radiography Database trial/'
in_path = '/data/CXR/Orignals/COVID-19 Radiography Database/COVID-19_Radiography_Dataset 3616/'

#out_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 3c/'
out_path = '/data/CXR/Training_split_2c/'+ str(dim) +'adap_70-40'


def get_folders_only(path):
    return next(os.walk(path))[1]
classes = get_folders_only(in_path)
#classes = ['COVID', 'Normal', 'Viral Pneumonia']
classes = ['COVID', 'Normal']

def make_split_dir_classwise(out_path, class_):
    train_path = out_path + '/train/'
    test_path =  out_path + '/test/'
    val_path =  out_path + '/val/'
    
    class_train_out = train_path + str(class_)  + '/'
    class_val_out = val_path+ str(class_)   + '/'
    class_test_out = test_path+ str(class_)  + '/'

    makemydir(class_train_out)
    makemydir(class_val_out)
    makemydir(class_test_out)
    return class_train_out, class_val_out, class_test_out
    

for n, class_ in tqdm(enumerate(classes), total=len(classes)):
    class_in_path = in_path + str(class_) + '/'
    ids = getImagesAndLabels(class_in_path)
    train_ids, test_val_ids  = train_test_split(ids, test_size=0.2, shuffle=True)
    val_ids, test_ids  = train_test_split(test_val_ids, test_size=0.5, shuffle=True)


    class_train_out, class_val_out, class_test_out = make_split_dir_classwise(out_path, class_)
    
    prep_cxr_classification(class_in_path, class_train_out, train_ids, model, dim)
    prep_cxr_classification(class_in_path, class_val_out, val_ids, model, dim)
    prep_cxr_classification(class_in_path, class_test_out, test_ids, model, dim)



def test_img(img_path):
    image = cv2.imread(img_path , 0)

    size = image.shape
    img = image
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    img = adap_equalize(img)
    img = np.expand_dims(np.array(img), axis = 0)
    preds = model.predict(img, verbose=1)
    preds_t = (preds > 0.5).astype(np.uint8)
    preds_t = np.squeeze(preds_t)
    #plt.imshow(preds_t)
    #plt.show()
    
    mask = resize(preds_t, size , mode='constant',  preserve_range=True)
    mask = (mask > 0).astype(np.uint8)
    plt.imshow(mask)
    plt.show()
    img_out, ratio = grabLungsbox(image, mask)
    
#img_path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset 3616/COVID/COVID-2521.png"
#test_img(img_path)

# =============================================================================
# # Things to change-
# # 1. in_path : dirrectory of the train/val data
# # 2. result_dir : No need to change. Based on os. getcwd
# # 3. batchsize
# # 4. Epoch
# # 5. modelname : only the unique name of the model to be added
# =============================================================================
   

# =============================================================================
# 1. in_path : directory of the original image datasets
# 2. out_path : directory of the output when a new folder will be created 
#               with a unique name which you need to decide.:
# 3. modelpath : path of the segementaion model to be loaded for mask predictions
# 4. post_process = True or False in case you want to post process the mask prdicitons
# 5. maskout option = True or False in the pre-processing function
# 6. Adaptive equalisation = 2nd last line of the pre-processing finction, option to use.
# 7. dim = Output dimensions of the pre-processing, you can also comment the resizing fucniton in the end of the pre-processing funciton incase you want the original size onlt.
# 
# =============================================================================

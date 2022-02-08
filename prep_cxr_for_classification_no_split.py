# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 01:18:05 2021

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
from methods_model_training import makemydir, load_img_fromIDs, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat, plot_test_maskout3, plot_maskout3, plot_maskonly2


import errno
from datetime import datetime

from sklearn.model_selection import train_test_split
import os.path


def prep_cxr_classification(in_path, out_dirr, ids, model):   
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
        if img.shape != (256, 256): #Input shape required for Mask prediction model
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
        #plt.imshow(mask)
        #plt.show()
        img_out, ratio = grabLungsbox(image, mask)
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
        #if img_out.shape != (dim, dim):
        #        img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_CUBIC)
        #img_out = adap_equalize(img_out)
    
        #img_out = cv2.resize(img_out, (dim, dim), interpolation = cv2.INTER_CUBIC) #for cxrs : INTER_CUBIC 
        cv2.imwrite(out_dirr  + name ,img_out)
        #x[n] = img_out  #Fill empty x with values from img
#    return x



modelpath = os.getcwd() +'/b32-e150_X5703_06-19__21-53-43/'
modelname = "256_b32-e150_X5703_Raw_blur_prep_model_lungs_segmtn_06-19__21-53-43"
# load model
yaml_file = open(modelpath + modelname + '.yaml', 'r')
model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(model_yaml)
# load weights into new model
model.load_weights(modelpath + modelname + '.h5')
print("Loaded model from disk")


        

#in_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/trail/cxrs/'
in_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19 Radiography Database trial/'
in_path = '/data/CXR/Orignals/COVID-19 Radiography Database/COVID-19_Radiography_Dataset 3616/'

out_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/trail/prepp/Training split/'
out_path = '/data/CXR/NoSplit_4c/Raw_cropped/'

dim = 512
def get_folders_only(path):
    return next(os.walk(path))[1]
classes = get_folders_only(in_path)


for n, class_ in tqdm(enumerate(classes), total=len(classes)):
    class_in_path = in_path + str(class_) + '/'
    ids = getImagesAndLabels(class_in_path)

    class_out_path = out_path + str(class_) + '/'
    makemydir(class_out_path)
    
    prep_cxr_classification(class_in_path, class_out_path, ids, model)


#ids = getImagesAndLabels(in_path)
#prep_cxr_classification(in_path, out_path, ids, model)

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


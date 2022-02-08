# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 04:42:46 2021

@author: AA086655
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:17:28 2021

@author: abhia
"""

import cv2
import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from varname import nameof
import glob
from tqdm import tqdm 

import keras
from keras.models import model_from_yaml
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

from Methods_img_proceess import prep_cxr_segmtn, adap_equalize
from methods_model_training import get_img_array, GradCam, makemydir,load_img_fromIDs, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels


import errno
from datetime import datetime



mydir = os.getcwd()+ "/LoadResults/"+datetime.now().strftime('%m-%d__%H-%M-%S') + "/"
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..


#initializing a random seed
seed = 38
np.random.seed = seed



modelpath = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Results_class/256_b64_X11008_ts06-29__21-01-30/'
modelname = "adap_class2_model256"

#modelpath = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Results2_class2/256_b64_e50X30720_ts07-05__20-30-33/'
#modelname = "adap_balancedClass2_more_variance_model2_256"

modelpath = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Resnet_Results_c2/256_b64_e50X20864_ts07-09__20-02-16"
modelname =  "/adap_resnet_256denseL_Class2_aug70-40_model3_optimze-BN-layer_256"

# load model
yaml_file = open(modelpath + modelname + '.yaml', 'r')
model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(model_yaml)
# load weights into new model
model.load_weights(modelpath + modelname + '.h5')
print("Loaded model from disk")

# summarize model.
model.summary()



       
def indpred(x, model):
    try:
        img_size = (model.input.shape[1], model.input.shape[1])
        #img = image.load_img(x ,target_size=input_shape) #image.load_img(x)
        preprocess_input = keras.applications.xception.preprocess_input
        #img =  preprocess_input(get_img_array(img_path, size=img_size))
        img = get_img_array(img_path, size=img_size)/255.0
        #img = np.asarray(img)
        #img = img.reshape([-1, model.input.shape[1], model.input.shape[2], model.input.shape[3]])
        #print(img.shape)
    except Exception as e:
        print(e)
        img, y_onehot  = x.next()
        y_int = [np.where(r==1)[0][0] for r in y_onehot] #converting one hot to integer
        print(y_int)
    plt.imshow(np.squeeze(img))
    Y_pred = model.predict(img)
    print('probabilities: ', model.predict(img))
    y_pred_t = np.argmax(Y_pred, axis=1)
    print('Predicted label: ', y_pred_t)
    #img = np.expand_dims(img, axis=0)

    try:
        class_dict = x.class_indices
        class_dict = dict(map(reversed, class_dict.items())) #reversing the dictionary mapping 
        #print(class_dict)
        print('Predicted class: ' , class_dict[y_pred_t[0]],)
        print('True class label: ' , class_dict[y_int[0]], y_onehot)
    except:
        pass

#indpred(test_set, model)
img_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c/train/COVID/COVID (5).png'
#indpred(img_path, model)
shape = model.input.shape[1]
print(shape)
#class_activation_map(model, img_path)
#GradCAM_out(model,img_path)




shape = model.input.shape[1]
img_size = (shape, shape)



path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c'
#path = '/data/CXR/Training_split_2c/512_4Xaug_adap_denoised/'
folder_name = os.path.basename(path)

train_path = path + '/train/'
val_path = path + '/val/'
test_path = path + '/test/'

# useful for getting number of output classes
#folders = glob(train_path + '/*')


test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function= None) #Put : adap_equalize for equalizing the inputs
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=img_size, batch_size=1, class_mode='categorical',interpolation = "bicubic")



img_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Seg Training split/train/prep_cxrs/CHNCXR_0171_0.png'
img_path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 3c/train/Viral Pneumonia/Viral Pneumonia (34).png'
img_path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c/train/COVID/COVID (34).png"
#img_path = "C:/Users/AA086655/Downloads/Suman_cxr.png"


shape = model.input.shape[1]
img_size = (shape, shape)

img = cv2.imread(img_path, 0)
img_out = adap_equalize(img)
if img_out.shape != img_size:
    interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>shape else  cv2.INTER_CUBIC
    img_out = cv2.resize(img_out, img_size, interpolation = interpolation_type)

#cv2.imwrite('Suman_cxr.png',img_out)
#img_path = 'Suman_cxr.png'

#indpred(img_path, model)
preds = GradCam(img_path, model,"conv5_block3_add",os.getcwd(), 1)
indpred(img_path, model)




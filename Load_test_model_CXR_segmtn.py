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
 
from tqdm import tqdm 

from keras.models import model_from_yaml

from Methods_img_proceess import prep_cxr_segmtn, morph
from methods_model_training import makemydir, load_img_fromIDs, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat, plot_test_maskout3, plot_maskout3, plot_maskonly2


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

# Assigninig Image width, hight and chanel(1 for Grayscale)
img_width = 256
img_hieght = 256
img_channels = 1

# Assigning Dataset paths
#path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19 Radiography Database 1200/COVID/'
path = "C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Seg Training split/"
test_path =  path #+ '/test/'

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'



try:
    test_ids = getImagesAndLabels(path + '/test/' + cxrs)
    x_test, y_test = load_img_n_masks_fromIDs(path + '/test/', test_ids, 256)
except:
    test_ids = getImagesAndLabels(test_path)# + cxrs)
    x_test = load_img_fromIDs(test_path, test_ids, 256)


modelpath = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Selected results/b16-e150_X5703_06-19__23-33-10/'
modelname = "256_b16-e150_X5703_Raw_blur_prep_model_lungs_segmtn_06-19__23-33-10"
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


preds_test = model.predict(x_testt, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_t = np.squeeze(preds_test_t)

# Prediction for a single image after required pre-processing of its data strcuture
preds_test = model.predict(np.expand_dims(np.array(x_test[1]), axis = 0), verbose=1)


# Perform a check on some random test samples
ix = random.randint(0, len(preds_test_t))x_testt[1]
grabLungsbox(x_test[ix], preds_test_t[ix])
plot_maskonly2(preds_test_t[ix], x_test[ix])
#plot_maskout3(preds_test_t[ix], x_test[ix], mydir, test_ids[ix] )
#grabLungsbox( x_test[ix], preds_test_t[ix])
#lot_test_maskout3(preds_test_t[ix],x_test[ix], y_test[ix], mydir, test_ids[ix] )



       
   


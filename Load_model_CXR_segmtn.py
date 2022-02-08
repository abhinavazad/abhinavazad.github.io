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
from methods_model_training import makemydir,load_img_fromIDs, grabLungsbox, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat, plot_test_maskout3, plot_maskout3


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
path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset 3616/Lung_Opacity/'
#train_path = path + '/train/'
#val_path =  path + '/val/'
test_path =  path #+ '/test/'

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'


# get the image labels out from the given foler path
#train_ids = getImagesAndLabels(train_path + cxrs)
#val_ids = getImagesAndLabels(val_path + cxrs) 
test_ids = getImagesAndLabels(test_path)# + cxrs)


'''
# For grabbing image IDs from csv file
colnames = ["Train", "Val","Test"]
data = pd.read_csv('train_test_val_ids.csv', names=colnames)
train_ids = data.Train.tolist()
val_ids = data.Val.tolist()
val_ids = [x for x in val_ids if str(x) != 'nan']
test_ids = data.Test.tolist()
test_ids = [x for x in test_ids if str(x) != 'nan']
'''


#x_train, y_train = load_img_n_masks_fromIDs(train_path, train_ids, 256)
#x_val, y_val = load_img_n_masks_fromIDs(val_path, val_ids, 256)
#x_test, y_test = load_img_n_masks_fromIDs(test_path, test_ids, 256)
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


preds_test = model.predict(x_test, verbose=1)
'''preds_train = model.predict(x_train, verbose=1)
preds_val = model.predict(x_val, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_train_t = np.squeeze(preds_train_t)

preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_val_t = np.squeeze(preds_val_t)
'''
preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_t = np.squeeze(preds_test_t)



#482 - contours detection problem

#results
#print("\ntest_res: ")
#test_res = confusion_mat(y_test, preds_test_t, nameof(preds_test_t), mydir)

'''
print("\nval_res: ")
train_res = confusion_mat(y_train, preds_train_t, nameof(preds_train_t), mydir)

print("\ntrain_res: ")
val_res = confusion_mat(y_train, preds_train_t, nameof(preds_train_t), mydir)


df1 = pd.DataFrame([test_res,val_res, train_res],
               index=["test","val","train"],
               columns=['IoU score', 'F1 score','Precision:','Sensitivity:','Specificity:','Accuracy:'])
df1.to_excel(mydir + "/output.xlsx")  
'''       


# Perform a sanity check on some random training samples
#ix = random.randint(0, len(preds_train_t))
#plot_test_maskout3(preds_train_t[ix],x_train[ix], y_train[ix],mydir, train_ids[ix])

# Perform a sanity check on some random validation samples
#ix = random.randint(0, len(preds_val_t))
#plot_test_maskout3(preds_val_t[ix],x_val[ix], y_val[ix], mydir, val_ids[ix] )

# Perform a check on some random test samples
ix = random.randint(0, len(preds_test_t))
print(ix)
plot_maskout3(preds_test_t[ix], x_test[ix], mydir, test_ids[ix] )
grabLungsbox( x_test[ix], preds_test_t[ix])
#plot_test_maskout3(preds_test_t[ix],x_test[ix], y_test[ix], mydir, test_ids[ix] )


       
   


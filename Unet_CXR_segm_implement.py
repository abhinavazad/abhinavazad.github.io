# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:04:03 2021

@author: abhia
"""

import cv2
import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from varname import nameof
from tqdm import tqdm 
import errno
from datetime import datetime
from contextlib import redirect_stdout

from methods_model_training import on_t_go_modelfit, makemydir, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat_seg, plot_test_maskout3
from Unet_Model_CXR_Lungs_seg import get_model

import time

       


# Assigninig Image width, hight and chanel(1 for Grayscale)
img_width = 256
img_height = 256
IMG_CHANNELS = 1

# Assigning Dataset paths
path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Seg Training split/'
train_path = path + '/train/'
test_path =  path + '/test/'
val_path =  path + '/val/'

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'



# USE WHEN TRAIN, VAL AND TEST DATASET ARE SEPERATE
# get the image labels out from the given foler path
train_ids = getImagesAndLabels(train_path + cxrs)
test_ids = getImagesAndLabels(test_path + cxrs)
val_ids = getImagesAndLabels(val_path + cxrs)


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

###############################

x_train, y_train = load_img_n_masks_fromIDs(train_path, train_ids, 256)
x_val, y_val = load_img_n_masks_fromIDs(val_path, val_ids, 256)
x_test, y_test = load_img_n_masks_fromIDs(test_path, test_ids, 256)

# Preprocessing for image_data_generator.fit function
x_train = np.expand_dims(np.array(x_train), axis = -1)
y_train = np.expand_dims(np.array(y_train), axis = -1)

x_val = np.expand_dims(np.array(x_val), axis = -1)
y_val = np.expand_dims(np.array(y_val), axis = -1)

x_test = np.expand_dims(np.array(x_test), axis = -1)
y_test = np.expand_dims(np.array(y_test), axis = -1)


# Get the model defined from get_model.py 
model = get_model(img_height,img_width,IMG_CHANNELS)
model.summary()



#Sanity check, view few mages
image_number = random.randint(0, len(x_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(x_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
#plt.savefig(result_dir + 'sampleCXR_Mask.png')
plt.cla()
plt.clf()



# Use this snippit for on-the-go data Augentation and auto validation slpit
#history, model = on_t_go_modelfit(model, x_train,y_train,x_val,y_val)

batch_size = 32
steps_per_epoch = (len(x_train))//batch_size
validation_steps = (len(x_val))//batch_size
epochs=200  #epoch= 50

result_dir = os.getcwd()+ "/Results/"+"_b"+ str(batch_size) + "_X"+ str(len(x_train)) + "_" + datetime.now().strftime('%m-%d__%H-%M-%S') +'/'
modelname = result_dir + str(img_width) +"_b"+ str(batch_size) + "_X"+ str(len(x_train)) + "_Raw_blur_prep_model_lungs_segmtn_" + datetime.now().strftime('%m-%d__%H-%M-%S')
makemydir(result_dir)

#Use Mode = max for accuracy and min for loss. 
checkpoint = ModelCheckpoint(result_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
callbacks_list = [checkpoint, early_stop, log_csv]

# #Modelcheckpoint
checkpoint = ModelCheckpoint(modelname + ".hdf5", monitor='val_acc', verbose=1, save_best_only=True,  mode='auto')
#This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
early_stop = EarlyStopping(monitor='val_acc', patience=2, min_delta=0.00001,  mode='auto', verbose=1, restore_best_weights = True)
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
callbacks_list = [checkpoint, early_stop, log_csv]

print('Lets start training.')
start = time.ctime()
#history = model.fit(x_train, y_train, validation_data=(x_val,y_val), steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs) 
history = model.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=epochs, batch_size = 2) 
end = time.ctime()
print('Training finished!')
print('====started: ', start, '--finished: ',end, ' ====')


   

################################


with open(result_dir + 'modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Saving the model to disk
# serialize model to YAML
model_yaml = model.to_yaml()
with open( modelname + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(modelname + ".h5")
print("Saved model to disk")



plot_model_hist(history, result_dir)


preds_train = model.predict(x_train, verbose=1)
preds_val = model.predict(x_val, verbose=1)
preds_test = model.predict(x_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)




#results
print("\ntest_res: ")
test_res = confusion_mat_seg(y_test, preds_test_t, nameof(preds_test_t), result_dir)

print("\nval_res: ")
val_res = confusion_mat_seg(y_val, preds_val_t, nameof(preds_val_t), result_dir)

print("\ntrain_res: ")
train_res = confusion_mat_seg(y_train, preds_train_t, nameof(preds_train_t), result_dir)


df1 = pd.DataFrame([test_res + [len(y_test)] +[batch_size, epochs, steps_per_epoch], val_res + [len(y_val)], train_res + [len(y_train)]],
               index=["test","val","train"],
               columns=['IoU score', 'F1 score','Precision:','Sensitivity','Specificity','Accuracy', 'Datapoints','Batchsize', 'Epochs', 'StepsPerEpoch'])
df1.to_excel(result_dir + "/output"+ datetime.now().strftime('%m-%d__%H-%M-%S') +".xlsx")  
       


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
plot_test_maskout3(preds_train_t[ix],x_train[ix], y_train[ix],result_dir, train_ids[ix])

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
plot_test_maskout3(preds_val_t[ix],x_val[ix], y_val[ix], result_dir, val_ids[ix] )

# Perform a check on some random test samples
ix = random.randint(0, len(preds_test_t))
plot_test_maskout3(preds_test_t[ix],x_test[ix], y_test[ix], result_dir, test_ids[ix] )



# =============================================================================
# # Things to change-
# # 1. in_path : dirrectory of the train/val data to be added
# # 2. result_dir : No need to change. based on os.getcwd
# # 3. modelname : path of the model with final name based on prep-process features
# # 4. Epoch
# # 5. batchsize
# =============================================================================
   

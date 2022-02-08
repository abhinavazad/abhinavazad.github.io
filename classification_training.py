# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:14:55 2021

@author: AA086655
"""

import cv2
import tensorflow as tf
import os
import numpy as np
import pandas as pd

import matplotlib
from sys import platform
if platform == "linux":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.models import load_model, Model, Sequential, model_from_yaml

from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda, Dense, Conv2D, MaxPool2D, Flatten, AveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras import optimizers
from keras.preprocessing import image

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers import RMSprop, Adam, SGD

from glob import glob
from datetime import datetime
from contextlib import redirect_stdout
import pandas
import seaborn as sns
from varname import nameof

from tqdm import tqdm
from skimage.transform import resize

from Methods_img_proceess import prep_cxr_segmtn, morph, adap_equalize
from methods_model_training import GradCam, append_multiple_lines, plot_class_confusion_matrix, eval, makemydir, load_img_fromIDs, load_img_n_masks_fromIDs, plot_model_hist, getImagesAndLabels, confusion_mat_seg, plot_test_maskout3, plot_maskout3, plot_maskonly2

import time
import collections


# =============================================================================
# 1. read ids
# 2. data path
# 3. Resize to 256 for getting mask predictions
# 4. Load segmentation model and get masks
# 5. Resize the mask to the original CXR img dimens : WHATSOEVER
# 4. Cropping RoI in the input CXR img about the lungs as per cnt detection in maks.
# 5. Resize the cropped CXR img to the input image dimens : 512
# 6. Data Aug
# 7. Denoising - optional preprocessing integrated in Data Aug
# 8. Feed in to the model
# 9. Validation metrices
# =============================================================================




def vgg16_model(IMAGE_SIZE, out_classes, modifiy):
    
    # Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
    # Here we will be using imagenet weights
    # Similary you can use the same template for Vgg 19, Resnet50, Mobilenet. All you have to import the library. Below are the examples
    vgg16 = VGG16(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)
    
    # don't train existing weights
    for layer in vgg16.layers:
        layer.trainable = False
    
    if modifiy == 0:
        # Flatenning outputs from the pretraineid model and adding a Softmax dense layer - you can add more if you want
        x = vgg16.layers[-2].output
        #x = vgg16.output
        x = Flatten()(x)
        prediction = Dense(len(out_classes), activation='softmax')(x)
    
    
    # create a model object
    #model = Model(inputs=vgg16.input, outputs=prediction)
    elif modifiy == 1: 
        new_model = vgg16.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(64, activation="relu")(new_model)
        new_model = Dropout(0.3)(new_model)
        prediction = Dense(len(out_classes), activation="softmax")(new_model)
    
    model = Model(inputs=vgg16.input, outputs= prediction)
    return model

def resnet_model(IMAGE_SIZE, out_classes, modifiy):
    # Import the ResNet50 library as shown below and add preprocessing layer to the front of VGG
    # Here we will be using imagenet weights
    # Similary you can use the same template for Vgg 19, Resnet50, Mobilenet. All you have to import the library. Below are the examples
    Resnet50 = ResNet50(input_shape=IMAGE_SIZE + [3],
                weights='imagenet', include_top=False)
    

    
    if modifiy == 0:
        # don't train existing weights
        for layer in Resnet50.layers:
            layer.trainable = False
        # Flatenning outputs from the pretraineid model and adding a Softmax dense layer - you can add more if you want
        x = Resnet50.layers[-2].output
        x = Flatten()(Resnet50.output)
        prediction = Dense(len(out_classes), activation='softmax')(x)
    
    
    # create a model object
    #model = Model(inputs=ResNet50.input, outputs=prediction)
    elif modifiy == 1: 
        
        for layer in Resnet50.layers:
            if isinstance(layer, keras.layers.normalization.BatchNormalization):
                layer.trainable = True #False for no optimzation 
            else:
                layer.trainable = False
        #new_model = Resnet50.output
        new_model = Resnet50.layers[-2].output
        new_model = AveragePooling2D(pool_size=(4, 4))(new_model) # size was (4,4) but thee were some erorrs coming with that
        new_model = Flatten(name="flatten")(new_model)
        new_model = Dense(512, activation="relu")(new_model)
        new_model = Dropout(0.3)(new_model)
        prediction = Dense(len(out_classes), activation="softmax")(new_model)
    
    model = Model(inputs=Resnet50.input, outputs= prediction)
    
    return model

    

# Assigninig Image width, hight and chanel(1 for Grayscale)
dim = 256

batch_size = 64
epochs= 50

img_width = dim
img_height = dim
IMG_CHANNELS = 1

# INPUT layer size, re-size all the images to this
IMAGE_SIZE = [img_width, img_height]


path = 'C:/Users/AA086655/OneDrive - Cerner Corporation/Desktop/Dataset/Training split 2c'
path = '/data/CXR/Training_split_2c/256adap_3Xbal_aug_70-40/'
folder_name = os.path.basename(path)
train_path = path + '/train/'
val_path = path + '/val/'
test_path = path + '/test/'



# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function= None) #Put : adap_equalize for equalizing the inputs
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function= None) #Put : adap_equalize for equalizing the inputs


# With on-the-go Augmentaion for training set
#train_set = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical')

# No on-the-go Augmentaion
train_set = test_datagen.flow_from_directory(train_path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical', interpolation = "bicubic")
val_set = test_datagen.flow_from_directory(val_path, shuffle=True,target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical',interpolation = "bicubic")
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical', interpolation = "bicubic")


steps_per_epoch = train_set.samples//train_set.batch_size #(len(x_train))//batch_size
validation_steps = val_set.samples//val_set.batch_size #(len(x_val))//batch_size


########################################################
# SETTING VARABLES
train_len = str(steps_per_epoch*batch_size) #str(len(x_train))
suffix = str(dim) +"_b"+ str(batch_size) +"_e"+ str(epochs) + "X"+ train_len + '_ts' + datetime.now().strftime('%m-%d__%H-%M-%S') 
result_dir = os.getcwd()+ "/VGG_results_c2_60-40/"+ suffix + '/'
makemydir(result_dir)
#keras_model_dir = result_dir + '/model_'  + str(img_width) +"_b"+ str(batch_size) + "_X"+ train_len+ '/'

modelname =  "/adap_vgg16(-2)_Class2_aug60-40_model0_" + str(dim)
modelpath = result_dir + modelname
#########################################################

#########################################################
# MODEL TRAINING 

# for getting number of output classes
folders = glob(train_path + '/*')

model = vgg16_model(IMAGE_SIZE, folders,0)
#model = resnet_model(IMAGE_SIZE, folders,1)

# view the final structure of the model
model.summary()

# view the trainable layers of the model
a =[]
for i, layers in enumerate(model.layers):
    a.append([i,layers.name, "-", layers.trainable])
    print(i,layers.name, "-", layers.trainable)


# tell the model what cost and optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


##Modelcheckpoint
checkpoint = ModelCheckpoint(modelpath + ".hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,  mode='max')
#This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
early_stop = EarlyStopping(monitor='val_accuracy', patience=3,  mode='max', verbose=1, restore_best_weights = True) #min_delta=0.5
#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(result_dir + 'my_logs.csv', separator=',', append=False)
callbacks_list = [checkpoint, early_stop, log_csv]


print('Lets start training.')
start = time.time()
# fit the model
history = model.fit(train_set, validation_data=test_set, epochs=epochs, steps_per_epoch= steps_per_epoch, validation_steps=validation_steps,callbacks=callbacks_list)
#history = model.fit(train_set, validation_data=test_set, epochs=epochs, batch_size = batch_size) #steps_per_epoch=len(train_set), validation_steps=len(val_set))
end = time.time()
print('Training finished!')
print('========Time taken: ', (end-start)/60,  'minutes ========')

################################################

################################################
# SAVING LOGS of training
with open(result_dir + 'modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        
with open(result_dir + 'modelsummary.txt', 'a') as output:
    for row in a:
        output.write(str(row) + '\n')
    
file = open(result_dir + 'modelsummary.txt', 'a')       
line = ['========Time taken for model training: ', str((end-start)/60),  'minutes ========',
        '\nTraining Dataset path: ', str(path), '\nTrainng Classes: ',str(train_set.class_indices), 
        '\nTotal train samples: ', (train_len),
        '\nClasswise train sample support: ', str(collections.Counter(train_set.labels)),
        '\nresult_dir path: ', str(result_dir), 
        '\nINPUT IMAGE_SIZE: ',str(IMAGE_SIZE), '\nEPOCHS: ',str(epochs), 
        '\nvalidation_steps: ',str(validation_steps), '\nTrain batch_size: ',
        str(batch_size), '\nsteps_per_epoch: ',str(steps_per_epoch)]

append_multiple_lines(result_dir + 'modelsummary.txt', line)

# Saving the model to disk
# serialize model to YAML
model_yaml = model.to_yaml()
with open( modelpath + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(modelpath + ".h5")
model.save(result_dir + 'saved_model')
print("Saved model to disk")

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
# save to csv: 
hist_csv_file =result_dir +  'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


try:
    plot_model_hist(history, result_dir)
except Exception as e:
    print("Exception due to ERROR: ", e)
    
##############################################################
# TESTING

# redefining Datagenerators with shuffle off
train_set = test_datagen.flow_from_directory(train_path, shuffle=False, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical',interpolation = "bicubic")
val_set = test_datagen.flow_from_directory(val_path, shuffle=False,target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical',interpolation = "bicubic")
test_set = test_datagen.flow_from_directory(test_path, shuffle=False, target_size=IMAGE_SIZE, batch_size=1, class_mode='categorical',interpolation = "bicubic")



eval(model, test_set, "test_set ", result_dir)
model.evaluate(test_set, steps = test_set.samples, verbose =1)
#model.predict_classes(test_set,batch_size)

eval(model, val_set, "val_set ", result_dir)
model.evaluate(val_set, steps = val_set.samples)


test_ids = getImagesAndLabels(test_path + '/COVID/')
preds = GradCam(test_path + '/COVID/' + test_ids[21] +'.png', model,"conv5_block3_out", result_dir, None)





# =============================================================================
# # Things to change-
# # 1. in_path : dirrectory of the train/val data
# # 2. result_dir :Rename the Top subfolder, rest is Based on os. getcwd
# # 3. batchsize
# # 4. Epoch
# # 5. modelname : only the unique name(based on feature and class) of the model to be added
# # 6. dim : input dimensions of the images for the training.
# =============================================================================
   
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:33:48 2021

@author: abhia
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
 
import matplotlib.pyplot as plt
from tqdm import tqdm 
from skimage.io import imread, imshow
import tensorflow as tf

import sys, os

from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.models import load_model, Model, Sequential, model_from_yaml
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
    
import pytesseract

import seaborn as sns
from varname import nameof


from Methods_img_proceess import prep_cxr_segmtn, morph, adap_equalize

import imutils

from PIL import Image, ImageFilter 
from random import randint
from keras import backend as K


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils

# Display
from IPython.display import Image, display
import matplotlib.cm as cm


def plot_class_confusion_matrix(cm, target_names, setname, result_dir):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    title=setname +' Confusion matrix',
    cmap=None
    normalize=False
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues') #'Spectral', 'Wistia', Pastel1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black") #"white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black") #"white" if cm[i, j] > thresh else "black")

    #sns.heatmap(cm, annot=True)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(result_dir + setname + 'class_confusion_mat.png')
    try:
        plt.show()
    except:
        pass


def eval(model, testdata, setname, result_dir):
    """
    This functions evaluates variaus evaluation metrices for the classification model for a given image data generator.
    Metrices includes: confusion matrix, Precison, Recall, Accuracy, 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    testdata : TYPE
        DESCRIPTION.
    setname : Prefix name for the dataset to be evaluated eg: test, train or val
    result_dir : Output directory

    Returns
    -------
    None.

    """
    class_dict = testdata.class_indices
    # + operator is used to perform task of concatenation
    class_dict = {setname + str(key): val for key, val in class_dict.items()}
    
    Y_pred = model.predict_generator(testdata, steps= testdata.samples)
    #print('1. ', Y_pred)
    y_pred_t = np.argmax(Y_pred, axis=1)
    #print('2. ', y_pred_t)
    cm = confusion_matrix(testdata.classes, y_pred_t)
    cm = np.transpose(cm)
    plot_class_confusion_matrix(cm,testdata.class_indices, setname, result_dir)

    print('cm:', cm)
    accuracy = accuracy_score(testdata.classes, y_pred_t)
    print("Accuracy in " + setname + " set: %0.1f%% " % (accuracy * 100))
    print(classification_report(testdata.classes, y_pred_t))
    report = classification_report(testdata.classes, y_pred_t, target_names= class_dict, output_dict=True)
    #print(report)
    try :
        import pandas
        df2 = pandas.DataFrame(report).transpose()
        df2.to_csv(result_dir +"eval_results" +".csv",mode='a')  
        print(df2)
    except Exception as e:
        print("Exception due to ERROR: ", e)
        df = pandas.DataFrame(report).transpose()
        df.to_csv(result_dir +"eval_results" +".csv")  
        print(df)
        #df.to_csv(result_dir + setname +"eval_results" +".csv") 

def union(a,b):
    """
    

    Parameters
    ----------
    a : 1st Recatangle's box coordinates as dereived form the ccv2.boundingRect() over the contours
    b : 2nd Rectangle's box coordinates as dereived form the ccv2.boundingRect() over the contours

    Returns
    -------
    x : Top-Left corner X coordinate
    y : Top-Left corner Y coordinate
    w : Final width of the Union rectongle
    h : Final height of the Union rectongle

    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def grabLungsbox(cxr, mask, maskout):
    """
    Outputs a cropped region around the lungs for a given CXR

    Parameters
    ----------
    cxr : Given CXR image whose lungs mask is determined
    mask : lungs mask as predicted by the lungs segmentation model
    maskout : Bool parameter, determines whether to output the whole region or only the Lung masked region in the cropped output

    Returns
    -------
    crop : Cropped Image around the lungs in the given CXR
    ratio : Ration of the area of the two lung contours detected by the segmentation model

    """
    if maskout == True:
        img = cxr*mask
    else:
        img = cxr
    ##Extracting contours out from the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    try: 
        #print(contours)
        a1 = cv2.contourArea(cnt[0])
        a2 = cv2.contourArea(cnt[1])
        ratio = a2/a1
        print("Contour Area ratio: ", ratio)
        show_img = cv2.drawContours(img.copy(), cnt, -1, (0, 255, 0),1)
        #plt.imshow(show_img, cmap='gray')
        
        x1,y1,w1,h1 = cv2.boundingRect(cnt[0])
        #cv2.rectangle(show_img,(x1,y1),(x1+w1,y1+h1),(255,255,255),2)
        x2,y2,w2,h2 = cv2.boundingRect(cnt[1])
        #cv2.rectangle(show_img,(x2,y2),(x2+w2,y2+h2),(255,255,255),2)
        
        x, y, w, h = union([x1,y1,w1,h1], [x2,y2,w2,h2])
        #cv2.rectangle(show_img,(x,y),(x+w,y+h),(0,0,0),2)
        #plt.imshow(show_img, cmap='gray')
        #plt.show
        crop = img[y:y+h, x:x+w]
        #plt.imshow(crop, cmap='gray')
        
        '''
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image with RoI')
        plt.imshow(show_img, cmap='gray')
        
        plt.subplot(232)
        plt.title('Cropped RoI')
        plt.imshow(crop, cmap='gray')
        '''
    except:
        crop = cxr
        ratio = 0
        #plt.title(str(ratio))
    return crop, ratio
    
def plot_maskout3(pred, x_img, mydir, id_):
    #pred = np.squeeze(pred)

    pred2 = morph(pred, 1)
    pred2 = morph(pred, 2)
    pred2 = morph(pred, 1)
    #maskout = x_img*pred
    
    plt.cla()
    plt.clf() 
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(x_img, cmap='gray')
    
    
    plt.subplot(232)
    plt.title('Prediction on test image')
    plt.imshow(x_img*pred, cmap='gray')
    
    plt.subplot(233)
    plt.title('Post-processed Prediction')
    plt.imshow(x_img*pred2, cmap='gray')
    plt.savefig(mydir + id_ +'_cxr_segmt.png')
    #plt.cla()
    #plt.clf()
    


def plot_test_maskout3(pred, x_img, y_truth, mydir, id_):
    #pred = np.squeeze(pred)
    '''
    pred = morph(pred, 1)
    pred = morph(pred, 2)
    pred = morph(pred, 1)
    #maskout = x_img*pred
    '''
    
    plt.cla()
    plt.clf() 
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(x_img, cmap='gray')
    
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(x_img*y_truth, cmap='gray')
    
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(x_img*pred, cmap='gray')
    plt.savefig(mydir + id_ +'_cxr_segmt.png')
    #plt.cla()
    #plt.clf()

def plot_maskonly2(pred, x_img):
    #pred = np.squeeze(pred)

    pred2 = morph(pred, 1)
    pred2 = morph(pred, 2)
    pred2 = morph(pred, 1)
    #maskout = x_img*pred
    
    plt.cla()
    plt.clf() 
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(x_img, cmap='gray')
    
    
    plt.subplot(232)
    plt.title('Mask output')
    plt.imshow(pred, cmap='gray')
    
    plt.subplot(233)
    plt.title('Post-processed mask')
    plt.imshow(pred2, cmap='gray')


    
    
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        # Now we are converting the PIL image into numpy array
        # getting the Id from the image
        Id = os.path.split(imagePath)[-1].split(".")[0]
        # extract the face from the training image sample
        Ids.append(Id)
    return Ids


def load_img_fromIDs_class(path, ids, img_width):
    # USE WHEN TRAIN, VAL AND TEST DATASET ARE SEPERATE
    # Get all the training images only assigned to x array
    cxrs = 'prep_cxrs/'
    lung_masks = 'LungsMasks/'
    img_height = img_width

    x = np.zeros((len(ids), img_height, img_width), dtype=np.uint8)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        img = cv2.imread(path + name, 0)#[:,IMG_CHANNELS] 
        img = adap_equalize(img)


        if img.shape != (img_height, img_width):
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x[n] = img  #Fill empty x with values from img
        
    print('Grabbing images in x array : Done!')
    return x


def load_img_fromIDs(path, ids, img_width):
    # USE WHEN TRAIN, VAL AND TEST DATASET ARE SEPERATE
    # Get all the training images only assigned to x array

    img_height = img_width

    x = np.zeros((len(ids), img_height, img_width), dtype=np.uint8)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        img = cv2.imread(path + name, 0)#[:,IMG_CHANNELS] 
        img = adap_equalize(img)

        if img.shape != (img_height, img_width):
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x[n] = img  #Fill empty x with values from img
        
    print('Grabbing images in x array : Done!')
    return x


def load_img_n_masks_fromIDs(path, ids, img_width):
    # USE WHEN TRAIN, VAL AND TEST DATASET ARE SEPERATE
    # Get all the training images and masks assigned to x and y array
    cxrs = 'prep_cxrs/'
    lung_masks = 'LungsMasks/'
    img_height = img_width

    x = np.zeros((len(ids), img_height, img_width), dtype=np.uint8)
    y = np.zeros((len(ids), img_height, img_width), dtype=np.bool)
    print('Grabbing images and masks') 
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        #print(name)
        img = cv2.imread(path + cxrs + name,0)#[:,IMG_CHANNELS] 
        # if the images need to be pre-processed!
        #img = prep_cxr_segmtn(img, img_hieght)
        if img.shape != (img_height, img_width):
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x[n] = img  #Fill empty x with values from img
        
        mask = np.zeros((img_height, img_width), dtype=np.bool)
        mask = imread(path + lung_masks + name)
        if mask.shape != (img_height, img_width):
            mask = resize(mask, (img_height, img_width), mode='constant', preserve_range=True)
        y[n] = mask   #Fill empty y with values from mask
    print('Grabbing images and masks in x and y : Done!')
    return x, y




def confusion_mat_seg(y_true, y_pred_t, setname, mydir):
    """
    

    Parameters
    ----------
    y_true : Ground truth mask 
    y_pred_t : Predicted mask
    setname : Name description for the confusion matrix
    mydir : Outout directory to save the cm image

    Returns
    -------
    None.

    """
    tp = np.logical_and(y_true==True, y_pred_t==True)
    tn = np.logical_and(y_true==False, y_pred_t==False)
    fp = np.logical_and(y_true==True, y_pred_t==False)
    fn = np.logical_and(y_true==False, y_pred_t==True)
    
    cmat = [[np.sum(tp), np.sum(fn)], [np.sum(fp), np.sum(tn)]]


    plt.figure(figsize = (6,6))
    plt.title(setname)
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1, linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(mydir + setname + 'confusion_mat_seg.png')
    try:
        plt.show()
    except:
        pass
    
    iou = np.sum(tp)/(np.sum(tp)+np.sum(fn)+np.sum(fp))
    f1 = (2*np.sum(tp))/((2*np.sum(tp))+np.sum(fn)+np.sum(fp))
    precision =  np.sum(tp)/(np.sum(tp)+np.sum(fp)) #pixcel_accuracy
    
    acc = (np.sum(tp)+np.sum(tn))/(np.sum(tp)+np.sum(tn)+np.sum(fn)+np.sum(fp))
    sensitivity= np.sum(tp)/(np.sum(tp)+np.sum(fn)) 
    specificity = np.sum(tn)/(np.sum(tn)+np.sum(fp)) 
    
    print('IoU score: ',iou,'\nF1 score:', f1,'\nPrecision:', precision,'\nSensitivity:', sensitivity,'\nSpecificity:', specificity,'\nAccuracy:', acc)
    return([iou, f1, precision, sensitivity, specificity, acc])
    
def iou(y_true, y_pred_t):
    y_pred=model.predict(x_test)
    y_pred_thresholded = y_pred > 0.5
    intersection = np.logical_and(y_true, y_pred_t)
    union = np.logical_or(y_true, y_pred_t)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU score is: ", iou_score)



def pixel_accuracy(y_true, y_pred_t):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    intersection = np.logical_and(y_true, y_pred_t)
    #print(np.sum(intersection))
    #print(np.sum(y_true==True))
    pixel_acc = np.sum(intersection)/np.sum(y_true==True)
    print("pixel accuracy score is: ", pixel_acc)


def plot_model_hist(history, mydir):
    #plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mydir + 'Train_val_loss.png')
    try:
        plt.show()
    except:
        pass

    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(mydir + 'Train_val_acc.png')
    try:
        plt.show()
    except:
        pass

def aug(img, des_path, total_image):
    '''
    This function Performs various Data Augmentaion for any given image.
    
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
    aug = ImageDataGenerator(
        rotation_range=3,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
    
    os.chdir(des_path)
    #img = cv2.imread(in_img_path,0)

    img_np = img_to_array(img)
    img_np = np.expand_dims(img_np, axis=0)
    image_gen = aug.flow(img_np, batch_size=20, save_to_dir=des_path,save_prefix="COVID", save_format="png")


    #total_image = 20
    i = 0
    for e in image_gen:
        if (i == total_image):
            break
        i = i +1

def denoise(img):
    img = np.squeeze(img)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.MedianFilter(size = 5)) 
    #img = ndimage.median_filter(img, size=4)
    #img = generic_filter(img, modal, (3, 3))
    img = np.expand_dims(img, axis = -1)
    return img


  
def aug_cxrs(id_, source_path, des_path, dim, n_times):
    '''
    This function Performs various Data Augmentaion for any given cxr image and masks together, the same way.
    
    It alters the following factors to augment any given image:
    rotation_range, zoom_range, width_shift_range, shear_range,
    height_shift_range, horizontal_flip, fill_mode.
    
    For X-ray horizontal_flip = True
    For OCR horizontal_flip = False
    
    It saves the Augmented images by randomly altering some features at "des_path"

    Parameters
    ----------
    id_ : id name of the cxr images which has to the same as the corresponding mask images in the masks folder
    in_img_path : Path of the source image file to be Augmented.
    des_path : Main directory Path where the augmented images are to be saved
    n_times : number of augmentations for a single image

    Returns
    -------
    No returns, automatically saves the augmented images at "des_path" directory.

    '''
    
    name = id_ + '.png'
    img = imread(source_path + name)
    
    if img.shape != (dim, dim):
        #print(img.shape, dim)
        interpolation_type =  cv2.INTER_AREA if img.shape[1]>dim else  cv2.INTER_CUBIC
        img = cv2.resize(img, (dim, dim), interpolation = interpolation_type)
    #img = adap_equalize(img)
    
    img = np.expand_dims(img_to_array(img), axis=0)
    
    seed = randint(0, 3000) #put any constand seed for same order

    img_data_gen_args = dict(rotation_range=2,
                     height_shift_range=0.08,
                     shear_range=0.5,
                     zoom_range=0.06,
                     horizontal_flip=True,
                     fill_mode="nearest", brightness_range=[0.7,1.3], 
                     preprocessing_function= None )#denoise ) #adap_equaliz


    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    #image_data_generator = ImageDataGenerator()

 
    makemydir(des_path)
    
    image_generator = image_data_generator.flow(img, seed=seed, batch_size=n_times, save_to_dir= des_path ,save_prefix= id_, save_format="png")

    
    #Total image augmentation output = iter
    iter = n_times
    i = 1
    for e in image_generator:
        if (i == iter):
            break
        i = i +1
            
        
        
def aug_cxr_n_mask(id_, source_path, des_path, n_times):
    '''

    This function Performs various Data Augmentaion for any given cxr image and masks together, the same way.
    
    It alters the following factors to augment any given image:
    rotation_range, zoom_range, width_shift_range, shear_range,
    height_shift_range, horizontal_flip, fill_mode.
    
    For X-ray horizontal_flip = True
    For OCR horizontal_flip = False
    
    It saves the Augmented images by randomly altering some features at "des_path"

    Parameters
    ----------
    id_ : id name of the cxr images which has to the same as the corresponding mask images in the masks folder
    in_img_path : Path of the source image file to be Augmented.
    des_path : Main directory Path where the augmented images are to be saved
    n_times : number of augmentations for a single image

    Returns
    -------
    No returns, automatically saves the augmented images at "des_path" directory.

    '''
    
    name = id_ + '.png'
    img = imread(source_path + 'prep_cxrs/' + name)
    mask = imread(source_path + 'LungsMasks/' + name)

    img = np.expand_dims(img_to_array(img), axis=0)
    mask = np.expand_dims(img_to_array(mask), axis=0)
    
    seed = 32
    img_data_gen_args = dict(rotation_range=3,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest", brightness_range=[0.7,1.3])
    
    mask_data_gen_args = dict(rotation_range=3,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest", brightness_range=[0.7,1.3],
                         preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
    
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    
    makemydir(des_path + '/prep_cxrs/')
    makemydir(des_path + '/LungsMasks/')
    
    image_generator = image_data_generator.flow(img, seed=seed, batch_size=4, save_to_dir=des_path + '/prep_cxrs/',save_prefix= id_, save_format="png")
    mask_generator = mask_data_generator.flow(mask, seed=seed, batch_size=4, save_to_dir=des_path + '/LungsMasks/',save_prefix= id_, save_format="png")
    
    
    #Total image augmentation output = iter
    iter = n_times
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

def mergeMasks():

    #lung_masks = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Montgomery/manual_mask/'
    lung_masks = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Lunds segmentation module/Model_Xrays/Masks/'
    
    train_ids = getImagesAndLabels(lung_masks) 
    
    x_train = np.zeros((len(train_ids), img_height, img_width), dtype=np.uint8)
    y_train = np.zeros((len(train_ids), img_height, img_width,1), dtype=np.bool)
    
    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    
        name = id_ + '.png'
        print(name)
        img = imread(cxrs + '/' + name)
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x_train[n] = img  #Fill empty x_train with values from img
        
        merged_mask = np.zeros((img_height, img_width), dtype=np.bool)
        mask1 = imread(lung_masks + '/leftMask/' + name)
        mask1 = np.expand_dims(resize(mask1, (img_height, img_width), mode='constant',  
                                          preserve_range=True), axis=-1)
        
    
        mask2 = imread(lung_masks + '/rightMask/' + name)
        mask2 = np.expand_dims(resize(mask2, (img_height, img_width), mode='constant',  
                                          preserve_range=True), axis=-1)
        merged_mask = np.maximum(mask1, mask2)  
        

        #Change to Output directory for the mearged masks
        #des_dir = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Lunds segmentation module/Model_Xrays/' + '516X516/'
        #makemydir(des_dir)
        #os.chdir(des_dir)

        #cv2.imwrite(name,merged_mask)
                
        y_train[n] = merged_mask   
        
        
def makemydir(path):
    '''
    This function attempts to make a directory for the given path else shows an error.

    Parameters
    ----------
    path : path for the directory which you wish to create.

    Returns
    -------
    If fails, returns : "OSError".

    '''
    if not os.path.exists(path):
        os.makedirs(path)

def on_t_go_modelfit(model,x_train,y_train,x_val,y_val):
    #Use this function for on-the-go data Augentation and auto validation slpit
    # Defining image data generator arguments
    # > New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
    # > Addition of preprocessing_function gives a binary mask rather than a mask with interpolated values. 
    # > Defining seed to makesure that same Augmentation happens with CXR images(X) and Maks(y)
    seed=24
    
    img_data_gen_args = dict(rotation_range=3,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest", brightness_range=[0.7,1.3])
    
    mask_data_gen_args = dict(rotation_range=3,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest", brightness_range=[0.7,1.3],
                         preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
    
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(x_train, augment=True, seed=seed)
    
    image_generator = image_data_generator.flow(x_train, seed=seed)
    valid_img_generator = image_data_generator.flow(x_val, seed=seed)
    
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_val, seed=seed)
    
    # CLubbing images and Masks together
    def my_image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)
    
    my_generator = my_image_mask_generator(image_generator, mask_generator)
    
    validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)
    
    
    x = image_generator.next()
    y = mask_generator.next()
    for i in range(0,1):
        image = x[i]
        mask = y[i]
        plt.subplot(1,2,1)
        plt.imshow(image[:,:,0], cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(mask[:,:,0])
        plt.show()
    
    batch_size = 16
    steps_per_epoch = 3*(len(x_train))//batch_size #data generator will generate extra images(3times) as we are puuting steps per epoch as 3 times the input dataset
    epochs=50
    history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=epochs)
    return history, model



def Load_prep_cxr_classification(in_path, ids, dim):
    
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
    
    x = np.zeros((len(ids), dim, dim), dtype=np.uint8)
    print('Grabbing images and masks')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        name = id_ + '.png'
        # if the images need to be pre-processed or resized!
        #img = prep_cxr_segmtn(path + name, img_width)
        image = cv2.imread(in_path + name, 0)  # [:,IMG_CHANNELS]
        size = image.shape
        img = image
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = adap_equalize(img)
        img = np.expand_dims(np.array(img), axis=0)
        preds = model.predict(img, verbose=1)
        preds_t = (preds > 0.5).astype(np.uint8)
        preds_t = np.squeeze(preds_t)
        plt.imshow(preds_t)
        plt.show()

        mask = resize(preds_t, size, mode='constant',  preserve_range=True)
        mask = (mask > 0).astype(np.uint8)
        plt.imshow(mask)
        plt.show()
        img_out = grabLungsbox(image, mask)
        # for cxrs : INTER_CUBIC
        interpolation_type =  cv2.INTER_AREA if img_out.shape[1]>dim else  cv2.INTER_CUBIC
        img_out = cv2.resize(img_out, (dim, dim),
                             interpolation = interpolation_type)
        x[n] = img_out  # Fill empty x with values from img
    return x



import keras
def get_img_array(img_path, size):
    # `img` is a PIL image of size of the input layer
    img = keras.preprocessing.image.load_img(img_path, target_size=size, interpolation='bicubic')
    # `array` is a float32 Numpy array of shape of the input layer
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
            # IF pred_index is not defined, make_gradcam_heatmap will automatically pick up the activation map for max predicted label using np.argmax()
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
 
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))



def GradCam(img_path, model, conv_layer, out_path, pred_index):
    # The convolution layer, whose activation is to be visulised using gradcam
    last_conv_layer_name = conv_layer #"block5_conv3"
    
    shape = model.input.shape[1]
    img_size = (shape, shape)


    # Remove last layer's softmax
    #model.layers[-1].activation = None
    
    img_array = get_img_array(img_path, size=img_size)/255.0
    #print(img_array) 
    preds = model.predict(img_array)
    print(preds)

    n = (len(np.asarray(preds)[0]))
    # IF pred_index is set, make_gradcam_heatmap will pick up the activation map for the given label only.
    for i in range(n):
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=i)
        save_and_display_gradcam(img_path, heatmap, out_path + "gradcam.jpg")
    return preds
   
def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)
     
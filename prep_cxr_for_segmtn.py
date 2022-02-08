# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:03:13 2021

@author: abhia
"""

import cv2
from tqdm import tqdm 
import pandas as pd

from Methods_img_proceess import createtrackbar, prep_cxr_segmtn, plotHist, getImagesAndLabels, aug, makemydir
from methods_model_training import mergeMasks



def write_in_path(ids, in_dirr, out_dirr):
    '''

    Parameters
    ----------
    id_ : ID labels list of the source images
    in_dirr : home directory of the images folders
    out_dirr : path for writing the processed images

    Returns
    -------
    Returns none but write the images in a new folder in the home diretory

    '''

    makemydir(out_dirr)
    print('Pre-processing and writing the CXRs images')
    i=1
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):   
    
        name = id_ + '.png'
        image_path = in_dirr + '/' + name 
        img = cv2.imread(image_path,0)
        #img = prep_cxr_segmtn(image_path, img_width)
        i=i+1     


 

img_width = 256
img_height = 256
img_channels = 1


in_path = '/home/aa086655/dataset/Raw256blurred/'
out_path = '/home/aa086655/dataset/AugRaw/Training split/'


cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'

#mergeMasks()  

#cxr_ids = getImagesAndLabels(in_path + cxrs)
#lung_masks_ids = getImagesAndLabels(in_path + lung_masks)

# For grabbing image IDs from csv file
colnames = ["Train", "Val","Test"]
data = pd.read_csv('train_test_val_ids.csv', names=colnames)
train_ids = data.Train.tolist()
val_ids = data.Val.tolist()
val_ids = [x for x in val_ids if str(x) != 'nan']
test_ids = data.Test.tolist()
test_ids = [x for x in test_ids if str(x) != 'nan']


#write_in_path(cxr_ids, in_path + cxrs, out_path+ cxrs)
#write_in_path(cxr_ids , in_path + lung_masks,out_path + lung_masks)


write_in_path(val_ids, in_path + cxrs, out_path + 'val/' + cxrs)
write_in_path(val_ids , in_path + lung_masks ,out_path + 'val/' + lung_masks)
write_in_path(test_ids, in_path + cxrs, out_path + 'test/' + cxrs)
write_in_path(test_ids , in_path + lung_masks, out_path + 'test/' + lung_masks)


'''
path_prep = os.path.join(path, 'Aug_trials' )
path =  path_prep + '/' + train_ids[2] + '.png'
##################
createtrackbar()
#While loop to refresh the output based on changing trackbar posiions
while(1):
    img = prep_cxr_segmtn(path, img_width)
    #cornerDetect(img,image)
    #segment(img,image)
    #ocr_core(img,image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
###################
'''

#aug_dir= path + 'Cxrs_preprocessed/aug/'
#makemydir(aug_dir)
#img = cv2.imread(path,0)
#aug( img ,aug_dir,20)


cv2.waitKey(0)
cv2.destroyAllWindows()
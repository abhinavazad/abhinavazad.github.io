# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:28:12 2021

@author: abhia
"""
import os
from tqdm import tqdm 

import MethodsDraft2_tuned
from MethodsDraft2_tuned import prep_cxr_segmtn, plotHist, getImagesAndLabels

cxr = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Schengzen/CXR_png'
lung_masks = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Schengzen/mask'


cxr_ids = getImagesAndLabels(cxr)
lung_masks_ids = getImagesAndLabels(lung_masks)


#crosscheck
j=0
for n, id_ in tqdm(enumerate(cxr_ids), total=len(cxr_ids)):
    mask_guess = cxr_ids[n] #+ '_' + 'mask'
    
    #cxr_guess = os.path.split(lung_masks_ids[n])[-1].split("_mask")[0]
    #print(mask_guess in lung_masks_ids)
    if mask_guess not in lung_masks_ids:
        print(j, '. ', mask_guess)
        j=j+1
print('total',j)

#otherway crosscheck
j=0
for n, id_ in tqdm(enumerate(lung_masks_ids), total=len(lung_masks_ids)):
    mask_guess = lung_masks_ids[n] #+ '_' + 'mask'
    
    #cxr_guess = os.path.split(lung_masks_ids[n])[-1].split("_mask")[0]
    #print(mask_guess in lung_masks_ids)
    if mask_guess not in cxr_ids:
        print(j, '. ', mask_guess)
        j=j+1
print('total',j)



    
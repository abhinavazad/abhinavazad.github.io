# -*- coding: utf-8 -*-
"""
Created on Tue May 18 18:33:03 2021

@author: abhia
"""

import os

cxr = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Schengzen/CXR_png'
lung_masks = 'D:/cerner/Datasets/CXR lungs Segmentation datasets/Schengzen/mask'

os.chdir(lung_masks)
for file in os.listdir(lung_masks):
    id_guess = os.path.split(file)[-1].split("_mask")[0] + '.png'
    #print(id_guess)
    os.rename(os.path.join(lung_masks, file), os.path.join(lung_masks, id_guess))

    
    
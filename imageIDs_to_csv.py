# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:39:35 2021

@author: abhia
"""

import csv
from itertools import zip_longest

from methods_model_training import getImagesAndLabels


# Assigning Dataset paths
path = 'D:/cerner/Datasets/Lungs Segementation training/trail set'
train_path = path + '/train/'
test_path =  path + '/test/'
val_path =  path + '/val/'

cxrs = 'prep_cxrs/'
lung_masks = 'LungsMasks/'


test_ids = getImagesAndLabels(test_path + cxrs)
train_ids = getImagesAndLabels(train_path + cxrs)
val_ids = getImagesAndLabels(val_path + cxrs)  


d = [train_ids, val_ids, test_ids]
export_data = zip_longest(*d, fillvalue = '')
with open('train_test_val_ids.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      #wr.writerow(("Train", "Val","Test"))
      wr.writerows(export_data)
myfile.close()



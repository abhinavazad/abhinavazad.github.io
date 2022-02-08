# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:06:43 2021

@author: AA086655
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

class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()
            
	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
                                        print(layer.name)
                                        return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
	def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
        
		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)

		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_JET):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)

def GradCAM_out(model,img_path):
    #tf.enable_eager_execution()
    shape = model.input.shape[1]
    
    # initialize the model to be VGG16
    #Model = model
    # check to see if we are using ResNet
    #if args["model"] == "resnet":
    #	Model = ResNet50
    # load the pre-trained CNN from disk
    #print("[INFO] loading model...")
    #model = Model(weights="imagenet")
    
    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = cv2.imread(img_path)
    #resized = cv2.resize(orig, (shape, shape))
    
    # load the input image from disk (in Keras/TensorFlow format) and
    # preprocess it
    image = load_img(img_path, target_size=(shape, shape))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    i = np.argmax(preds[0])
    # decode the ImageNet predictions to obtain the human-readable label
    #decoded = imagenet_utils.decode_predictions(preds)
    #(imagenetID, label, prob) = decoded[0][0]
    #label = "{}: {:.2f}%".format(label, prob * 100)
    #print("[INFO] {}".format(label))
    
    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
    
    # draw the predicted label on the output image
    #cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    #cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)
    plt.imshow(heatmap)
    #cv2.imshow("Output", output)
    #cv2.waitKey(0)
    

def class_activation_map(model, img_path):
    #tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.enable_eager_execution()
    dim = model.input.shape[1]
    #img_path =  "NORMAL2-IM-1440-0001.jpeg"
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = cv2.resize(img, (dim, dim))
    shape_in = [-1,model.input.shape[1], model.input.shape[2], model.input.shape[3]]
    print(shape_in)
    img = img.reshape(shape_in)
    print(img.shape)
    #img = np.expand_dims(img,axis=0)
    print(img.shape)
    
    predict = model.predict(img)
    target_class = np.argmax(predict[0])
    last_conv = model.get_layer('block5_pool')#('res5c_branch2c')
    grads =tf.GradientTape(persistent=False, watch_accessed_variables=True) #K.gradients(model.output[:,target_class],last_conv.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv.output[0]])
    pooled_grads_value,conv_layer_output = iterate([img])
    
    for i in range(conv_layer_output.output_shape): #check the dimenstions of the targetted conv_layer_output
        conv_layer_output[:,:,i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output,axis=-1)
    
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(heatmap)
    img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
    upsample = cv2.resize(heatmap, (256,256))
   
    output_path_gradcam = 'n13.jpeg'
    plt.imsave(output_path_gradcam,upsample * img_gray)
   

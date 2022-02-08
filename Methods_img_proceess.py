# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:29:55 2021

@author: abhia
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps

import time
import cProfile
import io
import pstats

import sys, os

import imutils
from imutils import perspective
from imutils import contours

from skimage.filters import threshold_otsu
from skimage.restoration import estimate_sigma
from skimage.transform import resize

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
    
import pytesseract


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
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


def profile(func):
    '''
    This function is for initiating the python profiler for 
    elemental-wise analysis of the overall runtime for the given
    function and its sub-functions
    
    Add @profile decorator above the any desired function to be analysed.

    Parameters
    ----------
    func : any desired function to be analysed.

    Returns
    -------
    wrapper : prints the run-time states for the given
    function and its sub-functions.

    '''
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    return wrapper


def nothing(x):
    '''
    trackbar callback fucntion does nothing but required for trackbar.

    Returns
    -------
    None.

    '''
    pass


def plotHist(image):
    '''
    Plots histogram of a given image using matplotlib

    Parameters
    ----------
    image : input binarized image

    Returns
    -------
    None.

    '''
    p1 = plt.hist(image)
    plt.show(p1)  
    plt.plot(cv2.calcHist([image],[0],None,[256],[0,256]))
    plt.show()

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



def estimate_noise(img):
    '''
    This Function estimates Sigma noise in any given image.

    Parameters
    ----------
    img : Deisred image whose signma noise had to be estmated.

    Returns
    -------
    sigma = Sigma noise index, higher the index, lower the noise.

    '''
    sigma = estimate_sigma(img, multichannel=True, average_sigmas=True)
    return sigma



def morph(img, switch):
    '''
    This function performs opening for an given image.
    Opening is essentially acheived by applying two filters:
        Erosion followed by dialiation
    
    1. Erosion : Expands the features removing noise
    2. Dialtion : Shrinks the feature and also useful in joining broken parts of an object
    
    In cases like noise removal, erosion is followed by dilation. 
    Because, erosion removes white noises, but it also shrinks our object. 
    So we dilate it. Since noise is gone, they won’t come back, but our object area increases.

    Parameters
    ----------
    img : Input Image
    morph_kernel : Kernel size for Erosion and Dilation.

    Returns
    -------
    opening : Openned image i.e smoothed and shrinked the expanded features of the image.

    '''
    morph_switch = 1

    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elpscekrnl1 =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    elpscekrnl2 =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #print("check51")
    #morph_switch = cv2.getTrackbarPos('morph_switch','edge_detect')
    
    #morph_kernel = 1 + 2*cv2.getTrackbarPos('morph_kernel','edge_detect')
    #print(cv2.getTrackbarPos('morph_kernel','edge_detect'))
    #kernel = np.ones((morph_kernel,morph_kernel), np.uint8)
    
    

    if switch == 1 and morph_switch:
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, elpscekrnl1)
        # opening = cv2.erode(img, kernel, iterations=1)
        # opening = cv2.dilate(img, kernel, iterations=1)
        return opening
    
    elif switch == 2 and morph_switch:
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, elpscekrnl2)
        return closing
    
    elif switch == 3 and morph_switch:
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
        return tophat

    else:

        return img


def contrast(input_Image):
    '''
    Pixel wise Multiplication of Image array with a factor changing the Contrast in the Image.

    Parameters
    ----------
    input_Image : Input image
    cntrst : multiplication factor*0.05.
    
    Returns
    -------
    img : input_Image*cntrst*0.05.

    '''
    #Assigning Contrast multiplying factor values based on Tracker’s position
    cntrst = cv2.getTrackbarPos('cntrst','trackbar') 
    c = (0.05)*cntrst
    img = (input_Image*c).astype('uint8')
    return img


def blur(img):
    '''
    This function performs blur filters over any given binarized image
    0. Mean blur: Mean averaging about neigbouring pixels
    1. Median blur: Median averaging about neigbouring pixels
    2. Gausian blur: Averaging based on weights of approximated Gausian Curve about neigbouring pixels
    3. Bilateral blur: Best filter for blurring with preserved edges or boundaring

    Parameters
    ----------
    img : Input birarized image
    blurr : Size of the Blur Kernel, not applicable for Bilateral filter blurring
    typ : switch between various blur filter -> "0" for Mean blur, 
    "1" for Median blur, "2" for Gaussian blur and "3" for Bilateral blur
    s : Only for Bil blur - increases the edge preservation limit.
        The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    d : Only for Bil. blur - diameter of the pixel neighbourhood

    Returns
    -------
    blurred_img : Final grayscale blurred image

    '''
    blurred_img = img 
    
    blurr = 3 #abs(-1 + 2*cv2.getTrackbarPos('blurr','blur_trackbar'))
    typ = 3 #cv2.getTrackbarPos('blur type','blur_trackbar')
    s = cv2.getTrackbarPos('s','blur_trackbar')
    d = cv2.getTrackbarPos('d','blur_trackbar')

    
        
    if typ == 1:
        blurred_img = cv2.blur(img, (blurr,blurr)).astype('uint8')
    elif typ == 2:
        blurred_img = cv2.medianBlur(img, blurr).astype('uint8')
    elif typ == 3:
        blurred_img = cv2.GaussianBlur(img, (blurr,blurr),0).astype('uint8')
    elif typ == 4:
        # Apply bilateral filter with d = 15, 
        # s = sigmaColor = sigmaSpace = 75.
        blurred_img = cv2.bilateralFilter(img, d, s, s)
    else :
        print("no blur: ", typ)
    #    blurred_img = img
    return blurred_img


def sharpen(img):
    '''
    switch to various Sharpening filters.

    Parameters
    ----------
    img : Binarized input image
    sharpen_kernel : Select kernel type
    N : Centre value of the Unsharp filter.

    Returns
    -------
    sharpened : Sharpened image

    '''
    #Assign Shparpening parameters based on trackbar position
    sharpen_switch = cv2.getTrackbarPos('sharpen_switch','sharpen_trackbar')
    N = cv2.getTrackbarPos('N','sharpen_trackbar')
    
    if sharpen_switch == 1: 
        
        Sharpening_kernel = np.array([[-1,-1,-1], 
                              [-1, 8,-1],
                              [-1,-1,-1]])*N/9
        sharpened = img + cv2.filter2D(img, -1, Sharpening_kernel)
        return sharpened
    elif sharpen_switch == 2:
        
        Sharpening_kernel = np.array([[-1,-2,-1], 
                              [-2, 12,-2],
                              [-1,-2,-1]])*N/16
        sharpened = img + cv2.filter2D(img, -1, Sharpening_kernel)
        return sharpened
    elif sharpen_switch == 3:
        
        Sharpening_kernel = np.array([[0,-1,0], 
                              [-1, 4,-2],
                              [0,-1,0]])*N/4 
        # sharpened = cv2.filter2D(img, -1, Sharpening_kernel) # if centre value is 8 
        sharpened = cv2.filter2D(img, -1, Sharpening_kernel) + img
        return sharpened
    elif sharpen_switch == 4:
        #N = 12, N=1 for laplacian filter
        Sharpening_kernel = np.array([[-1,-2,-1], 
                              [-2, N,-2],
                              [-1,-2,-1]])/16
        sharpened = img + cv2.filter2D(img, -1, Sharpening_kernel)
    else :
        return img



def fft(img):
    '''
    Performs fft filtering. 
    Conversion in frequency domain-> Filtering by drawing circles-> Invere Fourier Transform to visualise the image

    Parameters
    ----------
    img : Input binarized image
    r_out : Outer radius
    r_in : Inner radius

    Returns
    -------
    img_back : Output image after inver fft

    '''
    #Assigning FFT filter parameters for Inner and outer Radius based on Tracker’s position
    r_out = cv2.getTrackbarPos('fft_r_out','fft_trackbar')
    r_in = cv2.getTrackbarPos('fft_r_in','fft_trackbar')
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    res = np.hstack((img,magnitude_spectrum))
 


    #Band Pass filter  - Concentric circle mask, only the points living in concentric circle are ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 100
    #r_out = 50
    #r_in = 0
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_not(np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2), ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2)))

    #mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    

    mask[mask_area] = 1


    # apply mask and inverse DFT
    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


    return img_back

def HistEqualize(image, histequal_switch):
    '''
    Performs Histogram Equalization.

    Parameters
    ----------
    image : input binarized image.
    histequal_switch : Switches to the type of equalization. "1" for global equalization, "2" for adaptive equalized image

    Returns
    -------
    equalized : Equalized image

    '''
    #Assigning Histogram equalization choice values based on Tracker’s position
    try: 
        histequal_switch = cv2.getTrackbarPos('equalize\nhistogram','trackbar') 
    except:
        nothing

    ##for Normalized Equalization
    if histequal_switch == 1:
        equalized = cv2.equalizeHist(image)
    ##For adaptive equalization
    if histequal_switch == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(image)
    else:
        equalized = image
    
    return equalized

def plotHist(image):
    '''
    Plots histogram of a given image using matplotlib

    Parameters
    ----------
    image : input binarized image

    Returns
    -------
    None.

    '''
    p1 = plt.hist(image)
    plt.show(p1)  
    try:
        plt.plot(cv2.calcHist([image],[0],None,[256],[0,256]))
    except:
        nothing
    plt.show()
        
    
def createtrackbar():
    '''
    Creates the trackars for positional arguments with possibility of tuning and experimentation.

    Returns
    -------
    None.

    '''
    #Creating trackbar window
    cv2.namedWindow('trackbar')
    thres_switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar('thres_switch', 'trackbar',1,2,nothing)
    cv2.createTrackbar('Thres_blockSize','trackbar',9,30,nothing)
    cv2.createTrackbar('c_sub','trackbar',18,50,nothing)
    #create contrast tuning tarckbar
    cv2.createTrackbar('cntrst','trackbar',20,50,nothing) 
    #Histogram ealization selection
    cv2.createTrackbar('equalize\nhistogram','trackbar',2,2,nothing) 
      
      
    #create Blur control trackbar window
    cv2.namedWindow('blur_trackbar')
    cv2.createTrackbar('blurr','blur_trackbar',4,11,nothing)
    cv2.createTrackbar('blur type','blur_trackbar',4,4,nothing)
    cv2.createTrackbar('s','blur_trackbar',8,30,nothing) #60 for OCR
    cv2.createTrackbar('d','blur_trackbar',12,200,nothing) #15 for OCR
    
    # create Sharpeing control trackbar window
    cv2.namedWindow('sharpen_trackbar')
    #Shapening Kernel selection
    cv2.createTrackbar('sharpen_switch', 'sharpen_trackbar',0,3,nothing)
    cv2.createTrackbar('N','sharpen_trackbar',1,20,nothing) 
      
    '''
    # create FFT filter control trackbar window
    cv2.namedWindow('fft_trackbar')
    cv2.createTrackbar('fft_r_out','fft_trackbar',50,150,nothing) 
    cv2.createTrackbar('fft_r_in','fft_trackbar',0,150,nothing)     
    cv2.resizeWindow("fft_trackbar", 400,400)
    '''
    
    # create Edge detetection filter control trackbar window
    cv2.namedWindow('edge_detect')
    cv2.createTrackbar('threshold1','edge_detect',100,200,nothing) 
    cv2.createTrackbar('threshold2','edge_detect',250,300,nothing)   
    cv2.createTrackbar('aperture_size','edge_detect',1,3,nothing) 
    #create Morph switch for ON/OFF functionality
    morph_switch = '0 : OFF \n1 : Open \n2 : Close'
    cv2.createTrackbar('morph_switch', 'edge_detect',0,1,nothing)
    cv2.createTrackbar('morph_kernel','edge_detect',2,10,nothing)

    cv2.resizeWindow("trackbar", 500,500)
    cv2.resizeWindow("blur_trackbar", 400,400)
    cv2.resizeWindow("sharpen_trackbar", 400,400)

    cv2.resizeWindow("edge_detect", 500,500)
    


def edgedetect(img_binary):
    threshold1 = cv2.getTrackbarPos('threshold1','edge_detect')
    threshold2 = cv2.getTrackbarPos('threshold2','edge_detect')
    aperture_size = cv2.getTrackbarPos('aperture_size','edge_detect')
    ap_size = 3
    if aperture_size:
        ap_size =  abs( 1 + 2*aperture_size)
        
    morph_switch = cv2.getTrackbarPos('morph_switch','edge_detect')
    morph_kernel = 3
    if (cv2.getTrackbarPos('morph_kernel','edge_detect') and morph_switch):
        morph_kernel = 1 + 2*cv2.getTrackbarPos('morph_kernel','edge_detect')
    kernel = np.ones((morph_kernel,morph_kernel), np.uint8)
     
    edged = cv2.Canny(img_binary, threshold1, threshold2, (ap_size,ap_size))
    #edged = cv2.bitwise_not(edged).astype('uint8')
    cv2.imshow('edged image',edged)
    
    if morph_switch:
        edged = morph(edged, 2)

    cv2.imshow('filter  image',edged)
    
    return edged

def binarize(img):
    
    thres= img
    ###Assigning values based on Tracker’s position
    thres_switch = cv2.getTrackbarPos('thres_switch','trackbar')
    #Assign Kernel blocksize for Thresholding only if trackbar is active

    Thres_blockSize = 1 + 2*cv2.getTrackbarPos('Thres_blockSize','trackbar')
    #Assign Shift contstant value for Thresholding only if trackbar is active
    if cv2.getTrackbarPos('c_sub','trackbar'):
        c_sub = cv2.getTrackbarPos('c_sub','trackbar')
    elif thres_switch == 1:
        thres = cv2.adaptiveThreshold (img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Thres_blockSize, c_sub)

    #Binarizing  using Thresold_Otsu after a Gausian blur if thres_otsu variable is in ON position  
    elif thres_switch == 2:
        #thresh = threshold_otsu(image)
        #binary = image > thresh
        gaus = cv2.GaussianBlur(img,(3,3),0)
        ret3,thres = cv2.threshold(gaus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return thres


def add_border(input_image):
    

    border = int((input_image.shape[1])/10)
 
    bimg = cv2.copyMakeBorder(input_image, border, border, border, border, cv2.BORDER_CONSTANT, None, 0)
    bimg = bimg.astype('uint8')
    #cv2.imwrite('border.jpg', bimg)
    return bimg

def adap_equalize(img):
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    #print(img.shape, type(img))
    try: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass
    #img = np.squeeze(img)
    img = img.astype('uint8')
    img = clahe.apply(img)
    return img

def prep_cxr_segmtn(in_img_path, dim):
    
    blurr = abs(-1 + 2*cv2.getTrackbarPos('blurr','blur_trackbar'))
    typ = cv2.getTrackbarPos('blur type','blur_trackbar')
    s = cv2.getTrackbarPos('s','blur_trackbar')
    d = cv2.getTrackbarPos('d','blur_trackbar')
    

    IMG_WIDTH = dim
    IMG_HEIGHT = dim
    #Reading Input Image from the path "In_ing_path"
    image = cv2.imread(in_img_path,0)
    img = image
    #img = add_border(img)


    img = resize(img, (dim, dim), mode='constant', preserve_range=True) #for masks
    #img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_AREA) #for cxrs : INTER_CUBIC for upsizing and INTER_AREA for downsizing
    img = img.astype('uint8')
    
    # Appying Adaptive Histogram equalization on the image
    # clipLimit=3.5, tileGridSize=(8,8) for optimum.
    # As clipLimit increases Conrast ehancement limit increases.
    #img = adap_equalize(img
    #img = cv2.equalizeHist(img)
    
    #img = blur(img)
    
    '''
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(232)
    plt.title('After pre-processing')
    plt.imshow(img, cmap='gray')
    '''
    #res = np.hstack((image,img))
    #cv2.imshow('pre processed', res)
    
    return img


def processit_ocr(in_img_path):
    '''
    Main image pre-processing pipeline function which sub consists various 
    individual image processing function.

    Parameters
    ----------
    in_img_path : Input image path.

    Returns
    -------
    img : Final pre-processed image and the original image

    '''
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    
    #Reading Input Image from the path "In_ing_path"
    gray = cv2.imread(in_img_path,0)
    gray = gray.astype("uint8")
    #print(gray)

    #Set dimensions to resize
    dim = (600,400)
    
    #Resize to the new dimention using Inter_Area interpolation
    #gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA) 
    #image = imutils.resize(image, width=300)
    
    #Switching Colorspace to GRAYSCALE for image processing
    #Appying Histogram equalization on the image
    img = HistEqualize(gray,2)
    #img =  image

    #Appying Contrast on the image
    img= contrast(img)

    #Appying Sharpening on the image Sharpen_swtich variable is in ON position 
    img = sharpen(img) 
        
    #Appying Blur on the image 
    img = blur(img)

    tophat_segment(img)
    
    #Binarizing the image if thres_switch variable is in ON position  
    img = binarize(img)
    cv2.imshow('thres', img)

    '''
    ##Dilation and erosion of the image if morph_switch variable is in ON position  
    if morph_switch:
        img = morph(img, morph_switch)'''


    '''if sharpen_switch:
        img = sharpen(img)'''

    '''
    #Plot the Histrograms of the input and the processed image
    plotHist(image)
    plotHist(img)    
    '''

    #print('Sigma Noise estimation before and after: ', estimate_noise(image), estimate_noise(img))

    #Stacking the input and the pre-prosessed image
    res = np.hstack((gray,img))
    cv2.imshow('pre processed', res)
    
    #Showing results by the FFT filtering technique
    #img_back = fft(img, r_out, r_in)
    #cv2.imshow('Inverse', img_back )
    
    rotation_correc(img)
    #print(pytesseract.image_to_osd(img))

    return img, gray

def rotate(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def sum_rows(img):
    # Create a list to store the row sums
    row_sums = []
    # Iterate through the rows
    for r in range(img.shape[0]-1):
        # Sum the row
        row_sum = sum(sum(img[r:r+1,:]))
        # Add the sum to the list
        row_sums.append(row_sum)
    # Normalize range to (0,255)
    row_sums = (row_sums/max(row_sums)) * 255
    # Return
    return row_sums

def display_data(roi, row_sums, buffer):    
    # Create background to draw transform on
    bg = np.zeros((buffer*2, buffer*2), np.uint8)    
    # Iterate through the rows and draw on the background
    for row in range(roi.shape[0]-1):
        row_sum = row_sums[row]
        bg[row:row+1, :] = row_sum
    left_side = int(buffer/3)
    bg[:, left_side:] = roi[:,left_side:]   
    cv2.imshow('bg1', bg)
    k = cv2.waitKey(1)
    #out.write(cv2.cvtColor(cv2.resize(bg, (320,320)), cv2.COLOR_GRAY2BGR))
    return k

        
def rotation_correc(img):
    
    src = 255 - img
    scores = []
    
    h,w = src.shape
    small_dimention = min(h,w)
    src = src[:small_dimention, :small_dimention]
    
    out = cv2.VideoWriter('/home/stephen/Desktop/rotate.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'),
                      15, (320,320))

    # Rotate the image around in a circle
    angle = 0
    while angle <= 360:
        # Rotate the source image
        img = rotate(src, angle)    
        # Crop the center 1/3rd of the image (roi is filled with text)
        h,w = img.shape
        buffer = min(h, w) - int(min(h,w)/1.5)
        #roi = img.copy()
        roi = img[int(h/2-buffer):int(h/2+buffer), int(w/2-buffer):int(w/2+buffer)]
        # Create background to draw transform on
        bg = np.zeros((buffer*2, buffer*2), np.uint8)
        # Threshold image
        _, roi = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY)
        # Compute the sums of the rows
        row_sums = sum_rows(roi)
        # High score --> Zebra stripes
        score = np.count_nonzero(row_sums)
        if sum(row_sums) < 100000: scores.append(angle)
        k = display_data(roi, row_sums, buffer)
        if k == 27: break
        # Increment angle and try again
        angle += .5
    #cv2.destroyAllWindows()
    
    # Create images for display purposes	
    display = src.copy()
    # Create an image that contains bins. 
    bins_image = np.zeros_like(display)
    for angle in scores:
        # Rotate the image and draw a line on it
        display = rotate(display, angle)    
        cv2.line(display, (0,int(h/2)), (w,int(h/2)), 255, 1)
        display = rotate(display, -angle)
        # Rotate the bins image
        bins_image = rotate(bins_image, angle)
        # Draw a line on a temporary image
        temp = np.zeros_like(bins_image)
        cv2.line(temp, (0,int(h/2)), (w,int(h/2)), 50, 1)
        # 'Fill' up the bins
        bins_image += temp
        bins_image = rotate(bins_image, -angle)
    
    # Find the most filled bin
    for col in range(bins_image.shape[0]-1):
    	column = bins_image[:, col:col+1]
    	if np.amax(column) == np.amax(bins_image): x = col
    for col in range(bins_image.shape[0]-1):
    	column = bins_image[:, col:col+1]
    	if np.amax(column) == np.amax(bins_image): y = col
    # Draw circles showing the most filled bin
    cv2.circle(display, (x,y), 560, 255, 5)
    

    f, axarr = plt.subplots(1,3, sharex=True)
    axarr[0].imshow(src)
    axarr[1].imshow(display)
    axarr[2].imshow(bins_image)
    axarr[0].set_title('Source Image')
    axarr[1].set_title('Output')
    axarr[2].set_title('Bins Image')
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    plt.show()

def ocr_core(img, image):
    '''
    This function will handle the core OCR processing of images.

    Parameters
    ----------
    img : binarised and processed input image 
    image : Source image

    Returns
    -------
    text : text predicts from input image block

    '''
    
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    
    orig = image
    
    #image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    boxes = pytesseract.image_to_boxes(img) 
    
    #Drawing all the boxes on the processed image
    for b in boxes.splitlines():
        b = b.split(' ')
        orig = cv2.rectangle(orig, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (100, 255, 0), 1)
        
    cv2.imshow('OCR_Output', orig)
    
    ##Wroks only if Sliced parts of the image  texts are feeded
    text = pytesseract.image_to_string(img)
    print(text)
    
    return text

def tophat_segment(gray):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # apply a tophat (whitehat) morphological operator to find light, regions against a dark background
    tophat = morph(gray, 3)
    
    #cv2.imshow('tophat', tophat)
    
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    
    #cv2.imshow('gradX',gradX)
    
    gradX_closed = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    cv2.imshow('gradX_MORPH_CLOSE',gradX)
    thresh = cv2.threshold(gradX_closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #cv2.imshow('thresh otsu',thresh)
    
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    #cv2.imshow('thresh close',thresh_closed)
    
    all_together = np.hstack((gradX, gradX_closed, thresh_closed ))
    all_together = imutils.resize(all_together, width=900)
    cv2.imshow('tophat_mask: gradX, gradX_closed, thres_closed', all_together)
    
    return thresh_closed

def segment(img_binary, image):
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    orig = image
            
    edged = edgedetect(img_binary)
    
    #contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    
    draw_contours = cv2.drawContours(orig, contours, -1, (0, 255, 0),1)
    cv2.imshow('drawContours',orig)
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        #cv2.imshow('cnt',rect)
        
        ## Cropping the text block for giving input to OCR 
        #cropped = orig[y:y + h, x:x + w] 

        ## Apply OCR on the cropped image 
        #config = ('-l eng --oem 1 --psm 3')
        #text = pytesseract.image_to_string(cropped, config=config) 
        #print(text)
    try:
        cv2.imshow('cnt',rect)
    except:
        nothing
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    orig = image
            
    edged = edgedetect(img_binary)
    
    #contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    
    draw_contours = cv2.drawContours(orig, contours, -1, (0, 255, 0),1)
    cv2.imshow('drawContours',orig)
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        #cv2.imshow('cnt',rect)
        
        ## Cropping the text block for giving input to OCR 
        #cropped = orig[y:y + h, x:x + w] 

        ## Apply OCR on the cropped image 
        #config = ('-l eng --oem 1 --psm 3')
        #text = pytesseract.image_to_string(cropped, config=config) 
        #print(text)
    try:
        cv2.imshow('cnt',rect)
    except:
        nothing

def cornerDetect(gray, image):
    corners = cv2.goodFeaturesToTrack(gray,10,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        #cv2.circle(image,(x,y),3,255,-1)
    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]
        
    cv2.imshow('Corner Detect',image)

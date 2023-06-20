# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:36:58 2023

@author: user
"""

from PIL import Image
import os
import tensorflow as tf
from skimage.color import rgb2hsv
import cv2



#os.makedirs('LeafPicsDownScaled_hsv')
filelist = os.listdir("LeafPicsDownScaled")
for files in filelist:
    name = files.split('.')
    #img = Image.open('LeafPicsDownScaled\\'+files).convert('L')
    #img.save("LeafPicsDownScaledBW/" + name[0] + '_bw.tif')
    #files.close()
        








    # Read the image - Notice that OpenCV reads the images as BRG instead of RGB
    img = cv2.imread('LeafPicsDownScaled\\'+files)
    
    # Convert the BRG image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert the RGB image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = Image.fromarray(img)
    img.save("LeafPicsDownScaled_hsv/" + name[0] + '_hsv.tif')
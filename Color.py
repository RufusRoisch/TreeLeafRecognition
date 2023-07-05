# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:36:58 2023

@author: user
"""

# import packages

from PIL import Image
import os

# import tensorflow as tf
# from skimage.color import rgb2hsv
import cv2


# Change the color scheme of the images

# Create new folder for the images
os.makedirs("LeafPicsDownScaled_hsv", exist_ok=True)  # HSV images
os.makedirs("LeafPicsDownScaledBW", exist_ok=True)  # BW images

filelist = os.listdir("LeafPicsDownScaled")
for files in filelist:  # iterate through images
    name = files.split(".")  # split file name from extension
    img = Image.open("LeafPicsDownScaled\\" + files).convert(
        "L"
    )  # open and convert image to bw
    img.save("LeafPicsDownScaledBW/" + name[0] + "_bw.tif")  # save new image
    # files.close()

    # Convert Image to HSV Color Scheme
    img = cv2.imread(
        "LeafPicsDownScaled\\" + files
    )  # Read the image - Notice that OpenCV reads the images as BRG instead of RGB

    # Convert the BRG image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image from BGR to RGB

    # Convert the RGB image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert image from RGB to HSV
    img = Image.fromarray(img)  #
    img.save(
        "LeafPicsDownScaled_hsv/" + name[0] + "_hsv.tif"
    )  # save new image

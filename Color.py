# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:36:58 2023

@author: user
"""

from PIL import Image
import os
import tensorflow as tf
from skimage.color import rgb2hsv


filelist = os.listdir("LeafPicsDownScaled")
for files in filelist:
    name = files.split('.')
    img = Image.open('LeafPicsDownScaled\\'+files).convert('L')
    img.save("LeafPicsDownScaledBW/" + name[0] + '_bw.tif')
    #files.close()
    hsv_img = rgb2hsv(files)
    








# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:36:58 2023

@author: user
"""

from PIL import Image
import os
import tensorflow as tf



filelist = os.listdir("LeafPicsDownScaled")
for files in filelist:
    name = files.split('.')
    img = Image.open('LeafPicsDownScaled\\'+files).convert('L')
    img.save("LeafPicsDownScaled/" + name[0] + '_bw.tif')
    files.close()






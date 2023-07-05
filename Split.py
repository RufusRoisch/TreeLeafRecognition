# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:10:34 2023

@author: user
"""

import os
import random
import shutil
import re


# Split data and copy it to subdirectories for training and testing and each tree class

# Set the path to your dataset folder containing the images
dataset_path = "LeafPicsDownScaled"

# Set the ratio for splitting the data (e.g., 80% for training, 20% for testing)
train_ratio = 0.8
test_ratio = 0.2

# Set the paths for the train and test folders
train_folder = "train"
test_folder = "test"

# Create train and test folders
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get the list of image files in the dataset folder
image_files = [
    file for file in os.listdir(dataset_path) if file.endswith(".tif")
]

# Shuffle the image files randomly
random.shuffle(image_files)

# Split the dataset into train and test sets
train_size = int(train_ratio * len(image_files))
train_files = image_files[:train_size]
test_files = image_files[train_size:]

# Copy the train files to the train folder
train_label = {}
for file in train_files:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(train_folder, file)
    shutil.copyfile(src_path, dst_path)
    match = re.search(r"l(\d+)", file)
    if match:
        number = match.group(1)
        train_label[file] = number

# Create folder for each class within the train folder - important for the ImageDataGenerator
output_directory = "train"
os.makedirs(output_directory, exist_ok=True)
for key, value in train_label.items():
    # Check if the class value is present in the file name
    if value in key:
        # Create a directory for the class value if it doesn't exist
        class_directory = os.path.join(output_directory, value)
        os.makedirs(class_directory, exist_ok=True)

        # Copy the file to the corresponding class directory
        source_path = "train/" + key
        destination_path = os.path.join(class_directory, key)
        shutil.copyfile(source_path, destination_path)

# Copy the test files to the test folder
test_label = {}
for file in test_files:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(test_folder, file)
    shutil.copyfile(src_path, dst_path)
    match = re.search(r"l(\d+)", file)
    if match:
        number = match.group(1)
        test_label[file] = number

# Create folder for each class within the test folder - important for the ImageDataGenerator
output_directory = "test"
os.makedirs(output_directory, exist_ok=True)
for key, value in test_label.items():
    # Check if the class value is present in the file name
    if value in key:
        # Create a directory for the class value if it doesn't exist
        class_directory = os.path.join(output_directory, value)
        os.makedirs(class_directory, exist_ok=True)

        # Copy the file to the corresponding class directory
        source_path = "test/" + key
        destination_path = os.path.join(class_directory, key)
        shutil.copyfile(source_path, destination_path)

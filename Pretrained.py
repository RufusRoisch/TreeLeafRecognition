# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:28:34 2023

@author: user
"""

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras import layers
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

# Get the pretrained model and select an output layer
pre_trained_model = VGG19(input_shape=(300, 150, 3),
                          include_top=False,
                          weights='imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output


# Model definition

def get_model():
    # creates model

    x = layers.Flatten()(last_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(16, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    model.summary()

    # compiles model
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# used by get_data() to get a single picture out of /LeafPicsDownscaled
def get_picture(tree_class_num: int, leaf_num: int):
    # constructs path to picture from tree_num and leaf_num
    source_dir = "LeafPicsDownScaled"
    img_name = "l" + str(tree_class_num) + "nr" + f"{leaf_num:03d}" + ".tif"
    img_path = source_dir + "/" + img_name

    # loads the image from path as a numpy array and returns it
    img = np.array(Image.open(img_path))
    return img


# returns numpy arrays training_data, training_labels, test_data, test_labels
def get_data():
    # the time at which get_data() got called first
    start_time = time.time()

    # initializing data lists
    training_img_list = []
    training_label_list = []
    testing_img_list = []
    testing_label_list = []

    # iterates over all leaf pictures of every tree class
    for tree_class in range(1, 16, 1):
        for leaf_num in range(1, 76, 1):
            # gets next picture
            new_pic = get_picture(tree_class, leaf_num)

            # puts first 60 leafs and their label of each class into training data
            if leaf_num <= 60:
                training_img_list.append(new_pic)
                training_label_list.append(tree_class)
            # puts the last 15 leafs and their label of each class into testing data
            else:
                testing_img_list.append(new_pic)
                testing_label_list.append(tree_class)

            # measuring time taken for collecting data since start of get_data()
            print(f"Processing Data... \nTook {time.time() - start_time}s so far.")

    # returns collected and divided data
    return np.array(training_img_list), np.array(training_label_list), \
        np.array(testing_img_list), np.array(testing_label_list)


# Run the model

if __name__ == "__main__":
    # get training and test data
    training_images, training_labels, test_images, test_labels = get_data()
    print(f"Training-Data-Size: {len(training_images)} \nTest-Data-Size: {len(test_images)}")

    # turn values from 0-255 in images to a value from 0.0 - 1.0
    # which is easier for the neural network to understand
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # gets model
    model = get_model()

    # fits the data to the training data
    history = model.fit(training_images, training_labels, epochs=5)

    # evaluates the trained model using the test data
    print("Evaluation:")
    model.evaluate(test_images, test_labels)

    # Plot utility
    def plot_graphs(history, string):
      plt.plot(history.history[string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.show()

    # # Visualize the accuracy
    plot_graphs(history, 'loss')
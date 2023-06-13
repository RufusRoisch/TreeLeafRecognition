# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:15:18 2023

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:28:34 2023

@author: user
"""

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the pretrained model and select an output layer

pre_trained_model = VGG19(input_shape=(300, 150, 3),
                          include_top=False,
                          weights='imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('block2_pool')
last_output = last_layer.output


# Model definition

def get_model():
    # creates model

    x = layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 150, 3))(last_output)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 150, 3))(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 150, 3))(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(16, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    model.summary()

    # compiles model
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Data augmentation with ImageDataGenerator

# Set the paths for the train and test folders
train_folder = 'train'
test_folder = 'test'

# Add our data-augmentation parameters to ImageDataGenerator

training_images = ImageDataGenerator(rescale=1.0 / 255.,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True
                                     )

# Note that the validation data should not be augmented!
test_images = ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = training_images.flow_from_directory(train_folder,
                                                      batch_size=50,
                                                      class_mode='categorical',
                                                      target_size=(300, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_images.flow_from_directory(test_folder,
                                                       batch_size=50,
                                                       class_mode='categorical',
                                                       target_size=(300, 150))

# Run the model

# gets model
my_model = get_model()

# fits the data to the training data
history = my_model.fit(train_generator, validation_data=validation_generator,
                       steps_per_epoch=10,
                       epochs=5,
                       validation_steps=5,
                       verbose=1)

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

# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Model definition


def get_model():
    # creates model
    new_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(300, 150, 3)
            ),
            tf.keras.layers.MaxPooling2D(3, 3),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(300, 150, 3)
            ),
            tf.keras.layers.MaxPooling2D(3, 3),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(300, 150, 3)
            ),
            tf.keras.layers.MaxPooling2D(3, 3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.softmax),
        ]
    )

    # compiles model
    new_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    new_model.summary()

    # returns model
    return new_model


# Data augmentation with ImageDataGenerator

# Set the paths for the train and test folders
train_folder = "train"
test_folder = "test"

# Add our data-augmentation parameters to ImageDataGenerator

training_images = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Note that the validation data should not be augmented!
test_images = ImageDataGenerator(rescale=1.0 / 255)

# Flow training images in batches using train_datagen generator
train_generator = training_images.flow_from_directory(
    train_folder,
    batch_size=50,
    class_mode="categorical",
    target_size=(300, 150),
)

# Flow validation images in batches using test_datagen generator
validation_generator = test_images.flow_from_directory(
    test_folder,
    batch_size=50,
    class_mode="categorical",
    target_size=(300, 150),
)

# Run the model

# gets model
my_model = get_model()

# fits the data to the training data
history = my_model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=10,
    epochs=5,
    validation_steps=5,
    verbose=1,
)

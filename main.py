import time
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy


# used by get_data() to get a single picture out of /LeafPicsDownscaled
def get_picture(tree_class_num: int, leaf_num: int):
    # constructs path to picture from tree_num and leaf_num
    source_dir = "LeafPicsDownScaled"
    img_name = "l" + str(tree_class_num) + "nr" + f"{leaf_num:03d}" + ".tif"
    img_path = source_dir + "/" + img_name

    # loads the image from path as a numpy array and returns it
    img = numpy.array(Image.open(img_path))
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


# defines our neuralnetwork
def get_model():
    # creates model
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax)])

    # compiles model
    new_model.compile(optimizer=tf.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    new_model.summary()

    # returns model
    return new_model


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
    

# # Plot utility
# def plot_graphs(history, string):
#   plt.plot(history.history[string])
#   plt.xlabel("Epochs")
#   plt.ylabel(string)
#   plt.show()

# # Visualize the accuracy
# plot_graphs(history, 'accuracy')

# ignore pls
'''
index = 60
# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {new_label_array[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {new_img_array[index]}')

plt.imshow(training_img[index])
    plt.title(Labels.labels[training_label[index]])

    plt.show()
    '''

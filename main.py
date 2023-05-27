import time
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy

#used by get_data
def get_picture(tree_num: int, leaf_num: int):
    source_dir = "LeafPicsDownScaled"
    img_name = "l" + str(tree_num) + "nr" + f"{leaf_num:03d}" + ".tif"
    img_path = source_dir + "/" + img_name
    img = numpy.array(Image.open(img_path))
    return img

#gets training and test data + labels
def get_data():
    start_time = time.time()
    training_img_array = []
    training_label_array = []
    testing_img_array = []
    testing_label_array = []
    for tree_class in range(1, 16, 1):
        for leaf_num in range(1, 76, 1):
            new_pic = get_picture(tree_class, leaf_num)
            if leaf_num <= 60:
                training_img_array.append(new_pic)
                training_label_array.append(tree_class)
            else:
                testing_img_array.append(new_pic)
                testing_label_array.append(tree_class)
            print(f"Processing Data... \nTook {time.time() - start_time}s so far.")

    return np.array(training_img_array), np.array(training_label_array), np.array(testing_img_array), np.array(testing_label_array)


if __name__ == "__main__":
    training_images, training_labels, test_images, test_labels = get_data()
    print(f"Training-Data-Size: {len(training_images)} \nTest-Data-Size: {len(test_images)}")

    np.set_printoptions(linewidth=320)
    print(training_images[0])
    print(type(training_images[0]))
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(16, activation=tf.nn.softmax)])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=5)

    print("Evaluation:")
    model.evaluate(test_images, test_labels)

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

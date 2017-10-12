# import os
# import argparse
# from tqdm import tqdm

import numpy as np
import glob
# import pandas as pd
# from PIL import Image

import tensorflow as tf
from tensorflow.contrib.keras.api.keras.applications import VGG16
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dropout, Flatten, Dense, Input

# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
import numpy as np

"""
The GOAL is to create a cat-detector to prevent cats from doing their stuff in your garden.

The SYSTEM (hardware) setup should look like: a motion detection module, camera, processing unit
and sprinkler.

The PROCESS. A moving object enters the monitored area. The motion detector is activated and starts
the camera (either taking a photo or stream) The image is send to the processing unit and analyzed.
In case a cat is detected the sprinkler is started for a few seconds and the image is stored with
classification score. Otherwise only store the image with the classification score.

The MODEL APPROACH - Bottleneck method. Reuse the VGG16 network without the fully connected layers. Score from the 
new train set all images. The output off the last Convolutional Layer will be stored and usd as input for the custom
network. We build a new set of fully connected layers and train them by feeding the validation set to
"""


# Variable declaration

# train, validation and test paths
# Each folder contains a cats and dogs sub folder, containing the respective images
train_dir = "data/train"
validation_dir = "data/validation"
test_dir = "data/test_labeled"

# file names for the model weights
top_model_weights_path = 'bottleneck_fc_model.h5'

img_width, img_height = 150, 150
batch_size = 16
epochs = 50
nb_train_samples = 2000
nb_validation_samples = 400
nb_test_samples = 1000

# Set log level to INFO to get output while training.
tf.logging.set_verbosity(tf.logging.INFO)


def save_bottleneck_features():
    """
    Creates a scored dataset by feeding cats and dogs images to the pre-trained VGG16 network.
    The result is saved as a numpy file.
    :return: None
    """
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open("bottleneck_features_train.npy", 'wb'), bottleneck_features_train)
    print("Training samples saved.")

    generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print("Validation samples saved.")


def train_top_model():
    """
    Train two fully connected layers on top of the VGG16 network. As input the validation dataset
    is used.
    The weights of the top model are stored in a H5 file.
    :return: None
    """
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    # np.save(open('top_layer_model.json', 'w'), model.to_json())
    print("Top layers trained and saved.")


def get_performance():
    """Generate the model performance metrics for the trained model on the test dataset.
    :return:
    """
    # Load VGG16 model without FC layers
    # Add custom layers as used in the training phase
    # Load weights from H5 file
    # Combine VGG16 and custom model
    # Run model.evaluate(X_test, Y_test)

    # The base VGG model
    model_vgg16 = VGG16(include_top=False, weights='imagenet')
    print("VGG16 model loaded.")
    print(model_vgg16.summary())

    new_input = Input(shape=(200, 200, 3), name='image_input')

    output_model = model_vgg16(new_input)
    print(output_model)

    # The custom model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg16.layers[-1].output_shape[1:]))
    # top_model = model_vgg16.output
    # top_model = Flatten(name='Flatten')(top_model)
    top_model.add(Dense(256, activation='relu', name='FC1'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))
    # Load the weights
    top_model.load_weights(top_model_weights_path)
    # top_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print("Custom model defined and weights loaded.")

    # add the model on top of the convolutional base
    model = Model(input=model_vgg16.input, output=top_model)
    # print("Custom model added to top of VGG model.")

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for layer in model.layers[:25]:
    #     layer.trainable = False
    # print("First 25 layers of the model are set to fixed.")

    # Create scores on test data
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    scores = model.evaluate_generator(
        generator, nb_test_samples // batch_size)
    print(scores)


def score_vgg16(img_dir):
    """

    :param img_dir:
    :return:
    """
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        img_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    output = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print("VGG16 layers scored.")
    return output


def score_top_model(inputs):
    model = Sequential()
    model.add(Flatten(input_shape=inputs.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights(top_model_weights_path)
    print("Top model loaded.")
    predictions = model.predict_classes(inputs)
    print("Predictions made.")
    return predictions


def image_queue_test(path):
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    # print(tf.train.match_filenames_once("./data/test_labeled/cats/*.jpg"))
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path+"/*/*.jpg"))

    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)

    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        # tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get an image tensor and print its value.
        image_tensor = sess.run([image])
        print(image_tensor)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)


def main():
    # save_bottleneck_features()
    # train_top_model()
    # get_test_data()
    # image_queue_test(test_dir)
    vgg16_scores = score_vgg16(test_dir)
    pred = score_top_model(vgg16_scores)
    print(pred)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument("--inp", help="An input, any input.", type=str)
    # args = parser.parse_args()
    main()

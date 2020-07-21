# Connor Ragan
# cpragan@gmail.com 
# This file contains the Convolutional Neural Network for classifying the caliber of gunshot audio.
# The labels and feature data are in data.json

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# path to json file that stores MFCCs and caliber labels for each processed segment
DATA_PATH = "data.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # neccessary to add new axis to X train/test to match dimesionality of CNN
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    # build network topology
    model = keras.Sequential([

        # 1st conv layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        keras.layers.MaxPool2D((2,2), padding='same'),
        keras.layers.BatchNormalization(),

        # 2nd conv layer
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2,2), padding='same'),
        keras.layers.BatchNormalization(),

        # 3rd conv layer
        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.MaxPool2D((2,2), padding='same'),
        keras.layers.BatchNormalization(),

        # flatten output
        keras.layers.Flatten(),
        # fully connected layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        # 10 = possible target labels
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=40)
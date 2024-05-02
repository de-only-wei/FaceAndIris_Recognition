# Function to load and preprocess images

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize

input_shape = (128, 128, 3)  # Input image dimensions
num_classes = 25
batch_size = 32
epochs = 1
embedding_dim = 64  # Dimensionality of the embedding space

# Siamese network architecture
def create_siamese_network(input_shape, embedding_dim):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(embedding_dim)(x)
    return models.Model(input, output)

# Function to generate a single batch of pairs
def generate_single_batch(x_test, y_test, batch_indices):
    X = np.zeros((len(batch_indices), *input_shape))
    X2 = np.zeros((len(batch_indices), *input_shape))
    Y = np.zeros(len(batch_indices))

    for i, idx in enumerate(batch_indices):
        pair = x_test[idx]
        label = y_test[idx]
        X[i] = pair[0]
        X2[i] = pair[1]
        Y[i] = label

    return [X, X2], Y


def generate_batch(x_test, y_test, batch_size):
    while True:
        batch_indices = random.sample(range(len(x_test)), batch_size)
        yield generate_single_batch(x_test, y_test, batch_indices)

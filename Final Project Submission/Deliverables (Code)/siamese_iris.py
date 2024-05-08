import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

input_shape = (128, 128, 3)  # Input image dimensions
num_classes = 54
batch_size = 32
epochs = 1
embedding_dim = 64  # Dimensionality of the embedding space

# Function to load and preprocess images
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=input_shape[:2])
        img = img_to_array(img) / 255.0  # Normalize pixel values
        return img
    except Exception as e:
        print(f"Error loading image: {image_path}. {e}")
        return None

# Function to recursively search for image files in directories
def find_images(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images

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
def generate_single_batch(pairs, labels, batch_indices):
    X1 = np.zeros((len(batch_indices), *input_shape))
    X2 = np.zeros((len(batch_indices), *input_shape))
    Y = np.zeros(len(batch_indices))

    for i, idx in enumerate(batch_indices):
        pair = pairs[idx]
        label = labels[idx]
        img1 = preprocess_image(pair[0])
        img2 = preprocess_image(pair[1])
        X1[i] = img1
        X2[i] = img2
        Y[i] = label

    return [X1, X2], Y

# Function to generate batches of pairs with GPU acceleration
def generate_batch(pairs, labels, batch_size):
    while True:
        batch_indices = random.sample(range(len(pairs)), batch_size)
        yield generate_single_batch(pairs, labels, batch_indices)



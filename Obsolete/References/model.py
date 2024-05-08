import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg13_model(num_classes, leaky_relu_slope=0.1, dropout_rate=0.5):
    model = models.Sequential([
        # First block with 48 filters
        layers.SeparableConv2D(48, (3, 3), padding='same', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),

        # Second block with 96 filters
        layers.SeparableConv2D(96, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),

        # Third block with 192 filters
        layers.SeparableConv2D(192, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(192, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),

        # Fourth block with 384 filters
        layers.SeparableConv2D(384, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),

        # Final classification block
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

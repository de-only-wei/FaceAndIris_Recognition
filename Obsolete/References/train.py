import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import argparse
from model import build_vgg13_model
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_and_preprocess_data(csv_path, image_folder, image_size=(48, 48), num_classes=8):
    
    df = pd.read_csv(csv_path, header=None, usecols=range(10))  
    
    
    df.columns = ['filename', 'box', 'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    
    
    df['majority_vote'] = df.iloc[:, 2:].idxmax(axis=1)
    
    
    emotion_to_label = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7}
    df['label'] = df['majority_vote'].map(emotion_to_label)

    
    def parse_function(filename, label):
        filepath = tf.strings.join([image_folder, filename], separator='/')
        image_string = tf.io.read_file(filepath)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        image_resized = tf.image.resize(image_decoded, image_size)
        
        label_one_hot = tf.one_hot(label, depth=num_classes)
        
        return image_resized, label_one_hot

    
    dataset = tf.data.Dataset.from_tensor_slices((df['filename'].values, df['label'].values))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset



def build_and_compile_model(num_classes):
    model = build_vgg13_model(num_classes)  # from model.py
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Using 'categorical_crossentropy' for multi-class classification
                  metrics=['accuracy'])
    return model


@tf.function
def train_step(images, labels, model, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)


@tf.function
def val_step(images, labels, model, loss_fn, val_loss, val_accuracy):
    predictions = model(images, training=False)
    v_loss = loss_fn(labels, predictions)

    val_loss.update_state(v_loss)
    val_accuracy.update_state(labels, predictions)


def train_and_evaluate(base_folder, training_mode='majority', num_classes=8, max_epochs=25):
    best_val_loss = float('inf')

    
    train_image_folder = os.path.join(base_folder, 'FER2013Train')
    val_image_folder = os.path.join(base_folder, 'FER2013Valid')
    test_image_folder = os.path.join(base_folder, 'FER2013Test')

    train_csv_path = os.path.join(train_image_folder, 'label.csv')
    val_csv_path = os.path.join(val_image_folder, 'label.csv')
    test_csv_path = os.path.join(test_image_folder, 'label.csv')

    
    train_dataset = load_and_preprocess_data(train_csv_path, train_image_folder, image_size=(48, 48), num_classes=num_classes)
    val_dataset = load_and_preprocess_data(val_csv_path, val_image_folder, image_size=(48, 48), num_classes=num_classes)
    test_dataset = load_and_preprocess_data(test_csv_path, test_image_folder, image_size=(48, 48), num_classes=num_classes)

    #build model
    model = build_and_compile_model(num_classes)

    # Variables for summary after each epoch. Track learning progress.
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')


    # Start training loop
    for epoch in range(max_epochs):
       
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        epoch_val_loss = 0
        epoch_val_accuracy = 0
        train_batches = 0
        val_batches = 0

        # Training loop
        for images, labels in train_dataset:
            train_step(images, labels, model, loss_fn, optimizer, train_loss, train_accuracy)
            epoch_train_loss += train_loss.result().numpy()
            epoch_train_accuracy += train_accuracy.result().numpy()
            train_batches += 1
            

        # Validation loop
        for images, labels in val_dataset:
            val_step(images, labels, model, loss_fn, val_loss, val_accuracy)
            epoch_val_loss += val_loss.result().numpy()
            epoch_val_accuracy += val_accuracy.result().numpy()
            val_batches += 1

        #Quick check that we are indeed pulling data from dataset
        for images, labels in train_dataset.take(1):
            print(images.shape, labels.shape)

        
        epoch_train_loss /= train_batches
        epoch_train_accuracy /= train_batches
        epoch_val_loss /= val_batches
        epoch_val_accuracy /= val_batches

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save('./model/best_model.keras')
            print(f'Epoch {epoch+1}: New best model saved at val_loss {best_val_loss:.4f}')

        # Print metrics
        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy * 100:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy * 100:.2f}%")
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Reset metrics 
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()
        
    plot_loss(train_losses, val_losses)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_folder', type=str, required=True,
                        help='Base folder containing the training, validation, and testing data.')
    parser.add_argument('--training_mode', type=str, default='majority',
                        help="Specify the training mode: majority")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Maximum number of training epochs.")
    parser.add_argument('--num_classes', type=int, default=8,
                        help="Number of emotion classes.")

    args = parser.parse_args()
    train_and_evaluate(args.base_folder, args.training_mode, args.num_classes, args.epochs)



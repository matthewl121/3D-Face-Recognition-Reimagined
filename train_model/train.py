import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses TensorFlow log messages below error level

import numpy as np
import cv2
from glob import glob

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.applications import MobileNetV2  # Pre-trained model for feature extraction
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

# Global variables for image height, width, and number of landmarks
global image_h
global image_w
global num_landmarks

# Create directory if it doesn't already exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load paths for training, validation, and test datasets
def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "landmarks", "*.txt")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "landmarks", "*.txt")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "landmarks", "*.txt")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# Read and preprocess an image and its landmarks from file paths
def read_image_lankmarks(image_path, landmark_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load image
    h, w, _ = image.shape
    image = cv2.resize(image, (image_w, image_h))  
    image = image / 255.0 
    image = image.astype(np.float32)

    # Load landmarks from file and normalize coordinates
    data = open(landmark_path, "r").read()
    landmarks = []

    for line in data.strip().split("\n")[1:]:
        x, y = line.split(" ")
        x = float(x) / w  # Normalize x coordinate
        y = float(y) / h  # Normalize y coordinate
        landmarks.append(x)
        landmarks.append(y)

    landmarks = np.array(landmarks, dtype=np.float32)
    return image, landmarks

# Wrapper for preprocessing function in TensorFlow dataset
def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image, landmarks = read_image_lankmarks(x, y)
        return image, landmarks

    image, landmarks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([image_h, image_w, 3])  # Set shape for the image
    landmarks.set_shape([num_landmarks * 2])  # Set shape for the landmarks

    return image, landmarks

# Create TensorFlow dataset from file paths with batching and prefetching
def tf_dataset(x, y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess)
    ds = ds.batch(batch).prefetch(2)
    return ds

# Build a deep learning model for landmark detection based on MobileNetV2
def build_model(input_shape, num_landmarks):
    inputs = L.Input(input_shape)  # Define input layer

    # Load pre-trained MobileNetV2 for feature extraction
    backbone = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=0.5)
    backbone.trainable = True  # Fine-tune backbone

    x = backbone.output
    x = L.GlobalAveragePooling2D()(x)  # Pooling layer to reduce dimensions
    x = L.Dropout(0.2)(x)  # Dropout layer for regularization
    outputs = L.Dense(num_landmarks*2, activation="sigmoid")(x)  # Output layer with 2*num_landmarks outputs

    model = tf.keras.models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")  # Create a directory to store model files

    # Set image dimensions and number of landmarks
    image_h = 512
    image_w = 512
    num_landmarks = 500
    input_shape = (image_h, image_w, 3)
    batch_size = 32
    lr = 1e-3
    num_epochs = 100

    # Set dataset and model paths
    dataset_path = "img"
    model_path = os.path.join("files", "model.h5")

    # Load dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    # Create training and validation datasets
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    # Build and compile model
    model = build_model(input_shape, num_landmarks)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))

    # Set up training callbacks
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),  # Save best model
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),  # Adjust learning rate
        CSVLogger(csv_path, append=True),  # Log training progress to CSV file
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)  # Stop early if no improvement
    ]

    # Train the model
    model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )

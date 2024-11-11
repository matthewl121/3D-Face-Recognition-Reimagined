import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses TensorFlow warning messages

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm  # For displaying progress bars
import tensorflow as tf
from train import create_dir, load_dataset

global image_h
global image_w
global num_landmarks

def plot_lankmarks(image, landmarks):
    h, w, _ = image.shape 
    radius = int(h * 0.005)  

    for i in range(0, len(landmarks), 2):
        x = int(landmarks[i] * w)  # Scale x-coordinate
        y = int(landmarks[i+1] * h)  # Scale y-coordinate

        image = cv2.circle(image, (x, y), radius, (255, 0, 0), -1)

    return image 

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results")

    # Set image dimensions and number of landmarks
    image_h = 512
    image_w = 512
    num_landmarks = 500

    # Paths to dataset and model files
    dataset_path = "img"
    model_path = os.path.join("files", "model.h5")

    # Load dataset for training, validation, and testing
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")

    model = tf.keras.models.load_model(model_path)
  
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # Extract the filename without extension
        name = x.split("/")[-1].split(".")[0]
        
        # Load the test image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image_x = image  # Keep original image
        # Resize and normalize the image for model input
        image = cv2.resize(image, (image_w, image_h))
        image = image / 255.0 
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        # Load ground truth landmarks from annotation file
        data = open(y, "r").read()
        landmarks = []
        for line in data.strip().split("\n")[1:]:
            x, y = line.split(" ")
            # Normalize landmarks to image dimensions
            x = float(x) / image_x.shape[1]
            y = float(y) / image_x.shape[0]

            landmarks.append(x)
            landmarks.append(y)

        landmarks = np.array(landmarks, dtype=np.float32)

        # Make prediction using the model
        pred = model.predict(image, verbose=0)[0]
        pred = pred.astype(np.float32)

        # Plot ground truth and predicted landmarks on the image
        gt_landmarks = plot_lankmarks(image_x.copy(), landmarks)
        pred_landmarks = plot_lankmarks(image_x.copy(), pred)
        line = np.ones((image_x.shape[0], 10, 3)) * 255  # Separator line for visualization

        # Concatenate images with ground truth and predicted landmarks side-by-side
        cat_images = np.concatenate([gt_landmarks, line, pred_landmarks], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

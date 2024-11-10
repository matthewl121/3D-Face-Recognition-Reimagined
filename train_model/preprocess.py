import tensorflow as tf
from mtcnn import MTCNN
import numpy as np
import cv2

# Load pre-trained FaceNet model (you can download this model if needed)
facenet_model = tf.keras.models.load_model("facenet_keras.h5")
detector = MTCNN()

# Function to preprocess an image for FaceNet
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

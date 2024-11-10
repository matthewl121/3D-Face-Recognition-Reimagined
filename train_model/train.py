import cv2
import os
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np

face_embeddings = []
labels = []


facenet_model = tf.keras.models.load_model("facenet_keras.h5")
detector = MTCNN()

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

def get_embedding(image):
    face_pixels = preprocess_image(image)
    embedding = facenet_model.predict(face_pixels)[0]
    return embedding

for filename in os.listdir(r'C:\Users\mli00\Desktop\3D_Pictures'):
    img_path = os.path.join(r'C:\Users\mli00\Desktop\3D_Pictures', filename)
    image = cv2.imread(img_path)
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        face = image[y:y+height, x:x+width]
        embedding = get_embedding(face)
        face_embeddings.append(embedding)
        labels.append("Matt")

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet model for face embeddings
embedder = FaceNet()

# Function to preprocess face images for embedding extraction
def preprocess_face(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)

# Function to load annotated faces and labels from directories
def load_annotated_faces(image_dir, annotation_dir):
    data = []
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotation_dir, xml_file))
        root = tree.getroot()
        
        image_file = root.find('filename').text
        img_path = os.path.join(image_dir, image_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            # Extract bounding box coordinates for the face
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            # Crop face and add to dataset
            face = image[y_min:y_max, x_min:x_max]
            data.append((face, label))
    
    return data

# Load faces and labels from directories
faces, labels = zip(*load_annotated_faces(r"C:\Users\mli00\Desktop\3D_Pictures", r"landmarkXML"))

embeddings = []
for face in faces:
    # Extract embeddings
    detection = embedder.extract(face, threshold=0.95)
    if detection:
        embeddings.append(detection[0]['embedding'])

# Encode labels to integer format for the classifier
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train an SVM classifier using extracted embeddings and encoded labels
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, labels_encoded)

with open('classifier.pkl', 'wb') as f:
    pickle.dump((classifier, label_encoder), f)

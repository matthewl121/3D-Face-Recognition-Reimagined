import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load the classifier and label encoder
with open('classifier.pkl', 'rb') as f:
    classifier, label_encoder = pickle.load(f)

# Reload FaceNet model
facenet_model = load_model('facenet_keras.h5')

# Initialize MediaPipe Face Detection
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

# Initialize webcam
cap = cv2.VideoCapture(0)

def recognize_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            face = img[y1:y2, x1:x2]
            
            # Preprocess and predict
            face = preprocess_face(face)
            embedding = facenet_model.predict(face).flatten().reshape(1, -1)
            
            # Predict label with classifier
            probs = classifier.predict_proba(embedding)
            label_index = np.argmax(probs)
            label = label_encoder.inverse_transform([label_index])[0]
            confidence = probs[0][label_index]

            # Display bounding box and label on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({confidence*100:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Detect and recognize face in the frame
        frame = recognize_face(frame)
        
        # Display the frame
        cv2.imshow("Face Recognition", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
import argparse
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.dirname('util')))

from util.util_68 import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

REFERENCE_LANDMARKS_3D = {
    "left_eye_left_corner": np.array([36.0, 50.0, -10.0]),
    "left_eye_right_corner": np.array([39.0, 50.0, -10.0]),
    "right_eye_left_corner": np.array([42.0, 50.0, -10.0]),
    "right_eye_right_corner": np.array([45.0, 50.0, -10.0]),
    "nose_tip": np.array([30.0, 60.0, 0.0]),
    "nose_left": np.array([31.0, 60.0, -5.0]),
    "nose_right": np.array([35.0, 60.0, -5.0]),
    "mouth_left_corner": np.array([48.0, 85.0, -5.0]),
    "mouth_right_corner": np.array([54.0, 85.0, -5.0]),
    "chin": np.array([8.0, 90.0, -15.0]),
    "left_ear": np.array([0.0, 70.0, -15.0]),   # Approximate position for left ear
    "right_ear": np.array([80.0, 70.0, -15.0])  # Approximate position for right ear
}

def main(image_path):
    start_time = time.time()

    image = cv2.imread(image_path)
    load_time = time.time() - start_time 
    print(f"Time taken to load image: {load_time:.4f} seconds")

    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks_2d = extract_landmarks_2d(gray, face, predictor)

        candidate_landmarks_3d = map_2d_to_3d(landmarks_2d, REFERENCE_LANDMARKS_3D)

        similarity_score = recognize_face(REFERENCE_LANDMARKS_3D, candidate_landmarks_3d)
        print(f"Similarity Score: {similarity_score}")

        if similarity_score < 20:
            print("Face recognized")
        else:
            print("Face not recognized")

    # Display the image with landmarks
    for face in faces:
        for (x, y) in extract_landmarks_2d(gray, face, predictor):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
    # Draw the ear landmarks
    for ear_key in ["left_ear", "right_ear"]:
        x, y, _ = candidate_landmarks_3d[ear_key]
        cv2.circle(image, (int(x), int(y)), 4, (255, 0, 0), -1)  # Different color for ear

    cv2.imshow("Face Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect facial landmarks in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    main(args.image_path)

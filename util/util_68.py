import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import argparse

def extract_landmarks_2d(image, face, predictor):
    # Extracts 2D facial landmarks from the image
    landmarks = predictor(image, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return np.array(points)

def map_2d_to_3d(landmarks_2d, REFERENCE_LANDMARKS_3D):
    # Maps 2D landmarks to approximate 3D space based on reference
    mapped_landmarks_3d = {}
    for key, ref_point in REFERENCE_LANDMARKS_3D.items():
        # For ear landmarks, just directly map them, or you can adjust accordingly
        if key in ["left_ear", "right_ear"]:
            # Use a fixed mapping for ears
            x = landmarks_2d[3][0] if key == "left_ear" else landmarks_2d[13][0]  # Adjust according to ear landmarks
            y = landmarks_2d[3][1] if key == "left_ear" else landmarks_2d[13][1]
            z = ref_point[2]  # Z-depth from reference
            mapped_landmarks_3d[key] = np.array([x, y, z])
        else:
            idx = int(ref_point[0])  # Use the first value in `ref_point` as index
            x, y = landmarks_2d[idx]  # Corresponding 2D landmark
            z = ref_point[2]  # Z-depth from reference
            mapped_landmarks_3d[key] = np.array([x, y, z])
    return mapped_landmarks_3d

def recognize_face(reference_3d, candidate_3d):
    total_distance = 0.0
    for key in reference_3d:
        ref_point = reference_3d[key]
        cand_point = candidate_3d[key]
        total_distance += distance.euclidean(ref_point, cand_point)
    return total_distance
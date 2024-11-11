import cv2
import mediapipe as mp
import argparse
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def main(image_path):
    start_time = time.time()

    # read image from path
    image = cv2.imread(image_path)
    # track loading time
    load_time = time.time() - start_time 
    print(f"Time taken to load image: {load_time:.5f} seconds")

    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return
    
    # convert to RGB for mediapipe process
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect facial landmarks
    results = face_mesh.process(image_rgb)

    # check if facial landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    #Display facial features
    cv2.imshow('Face Mesh', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect facial landmarks using MediaPipe.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    main(args.image_path)

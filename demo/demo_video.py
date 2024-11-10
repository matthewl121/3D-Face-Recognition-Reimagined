import cv2
import mediapipe as mp
import time

try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Using GPU: {physical_devices[0].name}")
    else:
        print("Using CPU: No GPU found.")
except ImportError:
    print("TensorFlow is not installed, cannot check GPU availability.")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video source.")
            break
        
        frame_small = cv2.resize(frame, (800, 600))  
        image_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_small,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Green dots
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)  # Red lines
                )

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        latency = 1 / fps if fps > 0 else 0 

        cv2.putText(frame_small, f'FPS: {fps:.2f}', (frame_small.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_small, f'Latency: {latency:.2f}', (frame_small.shape[1] - 110, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Mesh', frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

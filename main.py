import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
import numpy as np
import threading
from mediapipe.framework.formats import landmark_pb2

# Global variables for sharing annotated frames between threads
latest_annotated_frame = None
frame_lock = threading.Lock()

# Visualization parameters
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_landmarks_on_image(result: vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame
    
    try:
        # Convert MediaPipe Image to numpy array (RGB format)
        annotated_image = image.numpy_view().copy()
        hand_landmarks_list = result.hand_landmarks or []
        handedness_list = result.handedness or []

        # Loop through detected hands
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw hand landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Draw handedness text
            if handedness:
                height, width, _ = annotated_image.shape
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coords) * width)
                text_y = int(min(y_coords) * height) - MARGIN
                
                cv2.putText(annotated_image, f"{handedness[0].category_name}",
                           (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                           FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Convert RGB to BGR for OpenCV and update shared frame
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        with frame_lock:
            latest_annotated_frame = annotated_image_bgr
            
    except Exception as e:
        print(f"Error in callback: {e}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Configure MediaPipe Hand Landmarker
model_path = 'models/hand_landmarker.task'  # Update this path
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=draw_landmarks_on_image,
    num_hands=2)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Perform async detection
        landmarker.detect_async(mp_image, timestamp_ms)

        # Display the latest annotated frame
        with frame_lock:
            display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
        
        cv2.imshow('Hand Landmarker', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
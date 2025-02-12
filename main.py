import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
import numpy as np
import threading
from mediapipe.framework.formats import landmark_pb2
import time
import queue

# Add frame queue and thread control
frame_queue = queue.Queue(maxsize=5)
should_stop = threading.Event()

def capture_frames(cap):
    while not should_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Clear queue to always have most recent frame
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
                
        frame_queue.put(frame)
    cap.release()

# Global variables for FPS calculation
prev_time = 0

# Shared data structures and locks
results_lock = threading.Lock()
original_frames = {}  # Stores original RGB frames by timestamp
face_results = {}     # Stores face landmarks by timestamp
hand_results = {}     # Stores hand landmarks and handedness by timestamp
latest_annotated_frame = None  # The combined frame to display

# Visualization parameters
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_face_landmarks_on_image(annotated_image, face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

def draw_hand_landmarks_on_image(annotated_image, hand_landmarks_list, handedness_list):
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx] if idx < len(handedness_list) else []

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

        if handedness:
            handedness_text = handedness[0].category_name
            height, width, _ = annotated_image.shape
            x_coords = [landmark.x for landmark in hand_landmarks]
            y_coords = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coords) * width)
            text_y = int(min(y_coords) * height) - MARGIN
            cv2.putText(annotated_image, f"{handedness_text}",
                       (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                       FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

def process_and_combine_landmarks(timestamp_ms):
    global latest_annotated_frame, original_frames, face_results, hand_results
    original_rgb = original_frames.pop(timestamp_ms, None)
    if original_rgb is None:
        return

    annotated_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    face_landmarks_list = face_results.pop(timestamp_ms, [])
    hand_data = hand_results.pop(timestamp_ms, ([], []))
    hand_landmarks_list, handedness_list = hand_data

    draw_face_landmarks_on_image(annotated_image, face_landmarks_list)
    draw_hand_landmarks_on_image(annotated_image, hand_landmarks_list, handedness_list)
    
    latest_annotated_frame = annotated_image.copy()

def face_landmark_callback(result: vision.FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global face_results
    with results_lock:
        face_results[timestamp_ms] = result.face_landmarks or []
        if timestamp_ms in hand_results:
            process_and_combine_landmarks(timestamp_ms)

def hand_landmark_callback(result: vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global hand_results
    with results_lock:
        hand_results[timestamp_ms] = (result.hand_landmarks or [], result.handedness or [])
        if timestamp_ms in face_results:
            process_and_combine_landmarks(timestamp_ms)

# Initialize webcam and start capture thread
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Reduce frame resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640/32)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480/32)

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
capture_thread.start()

# Configure MediaPipe Face and Hand Landmarker
face_model_path = 'models/face_landmarker.task'  # Update this path
hand_model_path = 'models/hand_landmarker.task'  # Update this path
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=face_landmark_callback,
    num_faces=2)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_landmark_callback,
    num_hands=2)

# Modify main processing loop
with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    while not should_stop.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        with results_lock:
            original_frames[timestamp_ms] = rgb_frame.copy()
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_landmarker.detect_async(mp_image, timestamp_ms)
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0
        prev_time = curr_time
        cv2.putText(display_frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Landmarker', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_stop.set()
            break

# Cleanup
should_stop.set()
if capture_thread.is_alive():
    capture_thread.join()
cv2.destroyAllWindows()
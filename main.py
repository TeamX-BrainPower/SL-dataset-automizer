import json
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2

# ---------------------------
# JSON Helper Functions
# ---------------------------
def create_top_json(word, total_frame_count, video_frame_rate):
    return {
        "word": word,
        "total_frame_count": total_frame_count,
        "video_frame_rate": video_frame_rate,
        "frameData": []
    }

def format_frame_info_into_json(hand_result, frame):
    """
    Build a JSON-friendly dict from the hand detection result.
    Expects hand_result.hand_landmarks to be a list of lists of landmarks,
    and hand_result.handedness to be a list of handedness objects (with a .category_name attribute).
    """
    frame_info = {"frame": frame, "hands": []}
    if hand_result and hand_result.hand_landmarks:
        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_info = {
                "handedness": handedness[0].category_name,
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks
                ]
            }
            frame_info["hands"].append(hand_info)
    return frame_info

def save_json_file(file_name, data):
    from os import makedirs
    makedirs('parsed-output', exist_ok=True)
    # Save the collected data to a JSON file in folder "parsed-output"
    with open(f'parsed-output/{file_name}.json', 'w') as f:
        json.dump(data, f, indent=4)

# ---------------------------
# Drawing Functions (unchanged)
# ---------------------------
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_face_landmarks_on_image(annotated_image, face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
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
        handedness = handedness_list[idx] if idx < len(handedness_list) else None

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        if handedness:
            handedness_text = handedness[0].category_name,
            height, width, _ = annotated_image.shape
            x_coords = [landmark.x for landmark in hand_landmarks]
            y_coords = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coords) * width)
            text_y = int(min(y_coords) * height) - MARGIN
            # cv2.putText(annotated_image, handedness_text,
            #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

# ---------------------------
# Main Processing
# ---------------------------
def main():
    # Video source and sign word
    sign_word = "abort"
    video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{sign_word}.mp4"
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get video info for JSON header (if available)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    json_builder = create_top_json(sign_word, total_frame_count, video_frame_rate)
    frame_count = 0

    # Configure MediaPipe Face and Hand Landmarker
    face_model_path = 'models/face_landmarker.task'  # Update this path if needed
    hand_model_path = 'models/hand_landmarker.task'  # Update this path if needed

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=2)
    
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2)

    # Initialize the landmark detectors using a context manager
    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and convert the frame for MediaPipe processing.
            flipped_frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

            # Get a timestamp in milliseconds.
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Create an MP Image and run detection (synchronously)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- JSON building from hand detection ---
            frame_info = format_frame_info_into_json(hand_result, frame_count)
            json_builder["frameData"].append(frame_info)
            frame_count += 1

            # Draw landmarks on the image.
            annotated_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if face_result and face_result.face_landmarks:
                draw_face_landmarks_on_image(annotated_image, face_result.face_landmarks)
            if hand_result and hand_result.hand_landmarks:
                draw_hand_landmarks_on_image(annotated_image, hand_result.hand_landmarks, hand_result.handedness)

            # Calculate and overlay FPS.
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0
            prev_time = curr_time
            cv2.putText(annotated_image, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display the annotated frame.
            cv2.imshow('Landmarker', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources and save JSON output.
    cap.release()
    cv2.destroyAllWindows()
    save_json_file(sign_word, json_builder)
    print(f"JSON data saved to parsed-output/{sign_word}.json")

if __name__ == "__main__":
    main()

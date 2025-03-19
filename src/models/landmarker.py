import mediapipe as mp
from mediapipe.tasks.python import vision
from contextlib import contextmanager

"""
The LandmarkerFactory class is responsible for creating the face, hand, gesture, and pose landmarkers. 
"""


class LandmarkerFactory:
    @staticmethod
    @contextmanager
    def create_landmarkers(config):
        base_options = mp.tasks.BaseOptions
        vision_mode = mp.tasks.vision.RunningMode

        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=config.face_model_path),
            running_mode=vision_mode.VIDEO,
            num_faces=config.num_faces
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=config.hand_model_path),
            running_mode=vision_mode.VIDEO,
            num_hands=config.num_hands
        )

        gesture_options = vision.GestureRecognizerOptions(
            base_options=base_options(model_asset_path=config.gesture_model_path), 
            running_mode=vision_mode.VIDEO
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=config.pose_model_path),
            running_mode=vision_mode.VIDEO
        )

        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options) 

        try:
            yield face_landmarker, hand_landmarker, gesture_recognizer, pose_landmarker
        finally:
            face_landmarker.close()
            hand_landmarker.close()
            gesture_recognizer.close()
            pose_landmarker.close()
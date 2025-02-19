from contextlib import contextmanager

import mediapipe as mp
from mediapipe.tasks.python import vision


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

        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        try:
            yield face_landmarker, hand_landmarker
        finally:
            face_landmarker.close()
            hand_landmarker.close()

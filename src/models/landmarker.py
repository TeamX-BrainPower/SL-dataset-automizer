import mediapipe as mp
from mediapipe.tasks.python import vision
from contextlib import contextmanager
from config import ProcessingConfig


class LandmarkerFactory:
    @staticmethod
    @contextmanager
    def create_landmarkers(config: ProcessingConfig):
        base_options = mp.tasks.BaseOptions

        if config.vision_mode == vision.RunningMode.LIVE_STREAM and (
            config.face_callback is None
            or config.hand_callback is None
            or config.pose_callback is None
        ):
            raise ValueError("Pose callbacks are None")

        face_model_file = open(config.face_model_path, "rb")
        face_model_data = face_model_file.read()
        face_model_file.close()

        hand_callback = None
        pose_callback = None
        face_callback = None

        if config.vision_mode == vision.RunningMode.LIVE_STREAM:
            hand_callback = config.hand_callback
            pose_callback = config.pose_callback
            face_callback = config.face_callback

        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_buffer=face_model_data),
            running_mode=config.vision_mode,
            num_faces=config.num_faces,
            result_callback=face_callback,
        )

        hand_model_file = open(config.hand_model_path, "rb")
        hand_model_data = hand_model_file.read()
        hand_model_file.close()

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_buffer=hand_model_data),
            running_mode=config.vision_mode,
            num_hands=config.num_hands,
            result_callback=hand_callback,
        )

        pose_model_file = open(config.pose_model_path, "rb")
        pose_model_data = pose_model_file.read()
        pose_model_file.close()

        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_buffer=pose_model_data),
            running_mode=config.vision_mode,
            result_callback=pose_callback,
        )

        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        try:
            yield face_landmarker, hand_landmarker, pose_landmarker
        finally:
            face_landmarker.close()
            hand_landmarker.close()
            pose_landmarker.close()

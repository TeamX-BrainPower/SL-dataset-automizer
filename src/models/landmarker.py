from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LandmarkerFactory:
    @staticmethod
    def create_hand_landmarker(config):
        base_options = python.BaseOptions(model_asset_path='src/models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO  # Set to video mode
        )
        return vision.HandLandmarker.create_from_options(options)

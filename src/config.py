from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    face_model_path: str = 'src/models/face_landmarker.task'
    hand_model_path: str = 'src/models/hand_landmarker.task'
    output_dir: str = 'data/raw'
    num_faces: int = 1
    num_hands: int = 2
    display_output: bool = True
    save_json: bool = True
    save_tfrecord: bool = False

@dataclass
class MotionDetectionConfig:
    threshold_delta = 0.000995 # Difference threshold between frames to detect motion
    detection_window_size = 5 # Number of frames to consider for motion detection
    missing_threshold = 10 # Number of frames before considering hand truly missing
    min_stable_frames = 10  # Minimum number of frames needed before checking for pauses

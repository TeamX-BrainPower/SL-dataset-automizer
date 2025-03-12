from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    face_model_path: str = "src/models/face_landmarker.task"
    hand_model_path: str = "src/models/hand_landmarker.task"
    gesture_model_path: str = "src/models/gesture_recognizer.task"
    pose_model_path: str = "src/models/pose_landmarker_lite.task"
    output_dir: str = "data/1-raw"
    num_faces: int = 1
    num_hands: int = 2
    display_output: bool = True
    save_json: bool = True
    save_tfrecord: bool = False

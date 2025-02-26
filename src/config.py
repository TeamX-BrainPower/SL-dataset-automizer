from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    face_model_path: str = 'src/models/face_landmarker.task'
    hand_model_path: str = 'src/models/hand_landmarker.task'
    output_dir: str = 'parsed-output'
    num_faces: int = 1
    num_hands: int = 2
    display_output: bool = True
    save_json: bool = True
    save_tfrecord: bool = False

from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    new_sign_recorder: bool = False

    face_model_path: str = 'models/face_landmarker.task'
    hand_model_path: str = 'models/hand_landmarker.task'
    output_dir: str = '../data/1-raw'
    num_faces: int = 1
    num_hands: int = 2
    display_output: bool = True
    save_json: bool = True
    save_tfrecord: bool = False

    recording_timeout = 2  # Max recording time in seconds
    movement_timeout = 0.5  # Time without movement to stop
    frame_size = (640, 480)  # Webcam resolution

    handle_data: bool = False
    fuzzing: bool = False
    merging: bool = False

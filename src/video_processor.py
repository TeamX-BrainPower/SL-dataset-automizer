from config import ProcessingConfig
from models.landmarker import LandmarkerFactory
from processor import Processor
import cv2 as cv
from time import time
import numpy as np


class VideoProcessor(Processor):
    files: list[str]
    headers: list[str]

    def __init__(self, config: ProcessingConfig, files: list[str]) -> None:
        super().__init__(config)
        self.files = files
        self.headers = ["timestamp"]

        extra = []

        for pose in [
            "nose",
            "right_shoulder",
            "left_shoulder",
            "right_elbow",
            "left_elbow",
            "right_wrist",
            "left_wrist",
        ]:
            for point in ["x", "y", "z"]:
                extra.append(f"{pose}_{point}")

        for hand in ["right", "left"]:
            for i in range(21):
                for point in ["x", "y", "z"]:
                    extra.append(f"{hand}_finger_{i}_{point}")

        self.headers += extra

        self.headers += [f"vel_{point}" for point in extra]
        self.headers += [f"acc_{point}" for point in extra]

        self.headers += [f"distance_{i}" for i in range(27)]

    def process(self) -> None:
        with LandmarkerFactory.create_landmarkers(self.config) as (
            self.face_landmarker,
            self.hand_landmarker,
            self.pose_landmarker,
        ):
            for video_file in self.files:
                cap = cv.VideoCapture(video_file)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config.cap_width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config.cap_height)

                if not cap.isOpened():
                    print("Could not open file:", video_file)
                    exit(1)

                # fps = cap.get(cv.CAP_PROP_FPS)
                # frame_rate = 1 / fps
                print("Processing file:", video_file)

                output_data = None
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp = time()

                    results, empty_hands, empty_pose = self.process_frame(
                        frame, timestamp, False
                    )

                    if self.config.pipeline:
                        output_data = self.config.pipeline.process(results)
                        if isinstance(output_data, np.ndarray):
                            print(output_data.shape)
                    # elapsed_time = time() - timestamp
                    # sleep_time = max(0, frame_rate - elapsed_time)
                    # sleep(sleep_time)

                if output_data is None:
                    print("Could not get any data for:", video_file)
                else:
                    print("output data shape:", output_data.shape)
                    file_path = (
                        f"{self.config.output_dir}/" + f"{video_file.split('.')[0]}.txt"
                    )

                    np.savetxt(
                        file_path,
                        output_data,
                        header=",".join(self.headers),
                        delimiter=",",
                    )
        return

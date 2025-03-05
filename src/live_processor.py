from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    PoseLandmarker,
    HandLandmarker,
)
from config import ProcessingConfig
from models.landmarker import LandmarkerFactory
import cv2 as cv
from time import time

from processor import Processor


class LiveProsessor(Processor):
    pose_landmarker: PoseLandmarker | None  # pyright: ignore
    hand_landmarker: HandLandmarker | None  # pyright: ignore
    face_landmarker: FaceLandmarker | None  # pyright: ignore
    cap: cv.VideoCapture
    config: ProcessingConfig

    def __init__(self, config: ProcessingConfig) -> None:
        super().__init__(config)
        self.cap = cv.VideoCapture(config.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.cap_height)
        self.cap.set(cv.CAP_PROP_FPS, config.fps_cap)

        return

    def process(self) -> None:
        with LandmarkerFactory.create_landmarkers(self.config) as (
            self.face_landmarker,
            self.hand_landmarker,
            self.pose_landmarker,
        ):
            last_timestamp = time()
            recording_index = 0
            recording_output = cv.VideoWriter(
                f"recording_{recording_index}.mp4",
                cv.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore
                10,
                (self.config.cap_width, self.config.cap_height),
            )

            cv.namedWindow("Test", cv.WINDOW_AUTOSIZE)

            while True:
                ret, frame = self.cap.read()
                show_frame = frame.copy()
                if not ret:
                    print("Could not get image")
                    break

                key = cv.waitKey(10)

                if key == 27:
                    break
                elif key == ord(" "):
                    if not self.config.recording:
                        print("Starting recording")
                        self.config.recording = True
                    else:
                        print("finished recording")
                        self.config.recording = False
                        recording_output.release()
                        recording_index += 1
                        recording_output = cv.VideoWriter(
                            f"{self.config.output_dir}/recording_{recording_index}.mp4",
                            cv.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore
                            10,
                            (self.config.cap_width, self.config.cap_height),
                        )

                timestamp = time()

                print("fps:", 1 / (timestamp - last_timestamp))

                if timestamp <= last_timestamp:
                    timestamp += 0.001

                last_timestamp = timestamp

                results, empty_hands, empty_pose = self.process_frame(
                    frame, timestamp, False
                )

                if self.config.pipeline:
                    self.config.pipeline.process(results)

                show_frame = cv.flip(show_frame, 1)
                cv.imshow("Test", show_frame)
                if self.config.recording:
                    recording_output.write(show_frame)

                if not empty_hands:
                    print("hands are empty")

                if not empty_pose:
                    print("pose is empty")
            recording_output.release()
            self.cap.release()
            cv.destroyAllWindows()

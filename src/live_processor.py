from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    PoseLandmarker,
    HandLandmarker,
)
from mediapipe import Image, ImageFormat
from config import ProcessingConfig
from models.landmarker import LandmarkerFactory
import cv2 as cv
import time


class LiveProsessor:
    pose_landmarker: PoseLandmarker | None  # pyright: ignore
    hand_landmarker: HandLandmarker | None  # pyright: ignore
    face_landmarker: FaceLandmarker | None  # pyright: ignore
    cap: cv.VideoCapture
    config: ProcessingConfig

    def __init__(self, config: ProcessingConfig) -> None:
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.face_landmarker = None
        self.config = config

        self.cap = cv.VideoCapture(config.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.cap_height)

        return

    def process_frame(self, frame: cv.typing.MatLike, timestamp: int):
        processed_frame = cv.flip(frame, 1)
        processed_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)

        self.hand_landmarker.detect_async(  # pyright: ignore
            mp_image, timestamp
        )
        self.pose_landmarker.detect_async(  # pyright: ignore
            mp_image, timestamp
        )

        return [], False  # results_arr, empty_frame

    def process(self) -> None:
        with LandmarkerFactory.create_landmarkers(self.config) as (
            self.face_landmarker,
            self.hand_landmarker,
            self.pose_landmarker,
        ):
            last_timestamp = 0
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("Could not get image")
                    break

                key = cv.waitKey(10)

                if key == 27:
                    break

                timestamp = int(time.time() * 1000)
                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1

                print("fps:", 1000 / (timestamp - last_timestamp))

                last_timestamp = timestamp

                results, empty_frame = self.process_frame(frame, timestamp)

                show_frame = frame.copy()
                cv.flip(show_frame, 1)
                cv.imshow("Test", show_frame)

                if empty_frame:
                    print("frame is empty")
                # else:
                #     print(results)

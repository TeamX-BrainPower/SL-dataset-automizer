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
import numpy as np
from numpy.typing import NDArray

from subscriber import Subscriber


class LiveProsessor:
    pose_landmarker: PoseLandmarker | None  # pyright: ignore
    hand_landmarker: HandLandmarker | None  # pyright: ignore
    face_landmarker: FaceLandmarker | None  # pyright: ignore
    cap: cv.VideoCapture
    config: ProcessingConfig
    subscribers: list[Subscriber]

    def __init__(self, config: ProcessingConfig) -> None:
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.face_landmarker = None
        self.config = config

        self.subscribers = []

        self.cap = cv.VideoCapture(config.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.cap_height)

        return

    def add_subscriber(self, subscriber: Subscriber):
        self.subscribers.append(subscriber)

    def process_frame(
        self, frame: cv.typing.MatLike, timestamp: int, world_coords: bool = False
    ) -> tuple[NDArray, bool, bool]:
        processed_frame = cv.flip(frame, 1)
        processed_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)

        hand_result = self.hand_landmarker.detect_for_video(  # pyright: ignore
            mp_image, timestamp
        )
        pose_result = self.pose_landmarker.detect_for_video(  # pyright: ignore
            mp_image, timestamp
        )

        results = [[-1, -1, -1]] * 49

        hands = False
        pose = False

        if not world_coords:
            if pose_result.pose_landmarks:
                pose = True
                poses = [
                    [pose.x, pose.y, pose.z]
                    for _, pose in enumerate(pose_result.pose_landmarks[0])
                ]
                results[0] = poses[0]
                results[1] = poses[12]
                results[2] = poses[11]
                results[3] = poses[14]
                results[4] = poses[13]
                results[5] = poses[16]
                results[6] = poses[15]

            if hand_result.hand_landmarks:
                hands = True
                for hand_landmarks, handedness in zip(
                    hand_result.hand_landmarks, hand_result.handedness
                ):
                    hand = handedness[0].category_name.lower()
                    if hand == "right":
                        results[7:28] = [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks
                        ]
                    elif hand == "left":
                        results[28:49] = [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks
                        ]

        else:
            if pose_result.pose_world_landmarks:
                pose = True
                poses = [
                    [pose.x, pose.y, pose.z]
                    for _, pose in enumerate(pose_result.pose_world_landmarks[0])
                ]
                results[0] = poses[0]
                results[1] = poses[12]
                results[2] = poses[11]
                results[3] = poses[14]
                results[4] = poses[13]
                results[5] = poses[16]
                results[6] = poses[15]

            if hand_result.hand_world_landmarks:
                hands = True
                for hand_landmarks, handedness in zip(
                    hand_result.hand_world_landmarks, hand_result.handedness
                ):
                    hand = handedness[0].category_name.lower()
                    wrist = np.array(
                        [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z]
                    )
                    coords = np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks
                        ]
                    )
                    coords -= wrist

                    if hand == "right":
                        results[7:28] = coords

                    elif hand == "left":
                        results[28:49] = coords

        return_arr = np.array(results)

        # compensate for the wrist
        if hands:
            return_arr[7:28] += return_arr[5]
            return_arr[28:49] += return_arr[6]

        return return_arr, hands, pose  # results_arr, empty_frame

    def process(self) -> None:
        with LandmarkerFactory.create_landmarkers(self.config) as (
            self.face_landmarker,
            self.hand_landmarker,
            self.pose_landmarker,
        ):
            last_timestamp = 0
            while True:
                ret, frame = self.cap.read()
                show_frame = frame.copy()
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

                results, empty_hands, empty_pose = self.process_frame(
                    frame, timestamp, True
                )

                for sub in self.subscribers:
                    sub.consume(results)

                cv.flip(show_frame, 1)
                # cv.imshow("Test", show_frame)

                if not empty_hands:
                    print("hands are empty")

                if not empty_pose:
                    print("pose is empty")
                # else:
                #     print(results)

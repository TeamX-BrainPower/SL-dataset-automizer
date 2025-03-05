from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    PoseLandmarker,
    HandLandmarker,
)
from abc import ABC
import mediapipe as mp
import numpy as np
from config import ProcessingConfig
import cv2 as cv
from mediapipe.tasks.python.vision import RunningMode


class Processor(ABC):
    config: ProcessingConfig
    pose_landmarker: PoseLandmarker | None  # pyright: ignore
    hand_landmarker: HandLandmarker | None  # pyright: ignore
    face_landmarker: FaceLandmarker | None  # pyright: ignore

    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.face_landmarker = None

    def process_video(
        self, image: mp.Image, timestamp: int, world_coords: bool = False
    ) -> tuple[np.ndarray, bool, bool]:
        results = np.empty((49, 3), dtype=np.float64)
        results[:] = np.nan
        hand_result = self.hand_landmarker.detect_for_video(  # pyright: ignore
            image, timestamp
        )
        pose_result = self.pose_landmarker.detect_for_video(  # pyright: ignore
            image, timestamp
        )

        pose = False
        hands = False

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
                    wrist = np.array(
                        [
                            hand_landmarks[0].x,
                            hand_landmarks[0].y,
                            hand_landmarks[0].z,
                        ]
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

        else:
            if pose_result.pose_world_landmarks:
                pose = True
                poses = [
                    [pose.x, pose.y, pose.z]
                    for _, pose in enumerate(pose_result.pose_world_landmarks[0])
                ]
                results[0] = poses[0]  # nose
                results[1] = poses[12]  # right shoulder
                results[2] = poses[11]  # left shoulder
                results[3] = poses[14]  # right elbow
                results[4] = poses[13]  # left elbow
                results[5] = poses[16]  # right wrist
                results[6] = poses[15]  # left wrist

            if hand_result.hand_world_landmarks:
                hands = True
                for hand_landmarks, handedness in zip(
                    hand_result.hand_world_landmarks, hand_result.handedness
                ):
                    hand = handedness[0].category_name.lower()
                    wrist = np.array(
                        [
                            hand_landmarks[0].x,
                            hand_landmarks[0].y,
                            hand_landmarks[0].z,
                        ]
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

        return results, pose, hands

    def process_live_stream(self, image: mp.Image, timestamp: int) -> None:
        self.hand_landmarker.detect_async(  # pyright: ignore
            image, timestamp
        )
        self.pose_landmarker.detect_async(  # pyright: ignore
            image, timestamp
        )
        return

    def process_image(
        self, image: mp.Image, world_coords: bool = False
    ) -> tuple[np.ndarray, bool, bool]:
        results = np.empty((49, 3), dtype=np.float64)
        results[:] = np.nan
        hand_result = self.hand_landmarker.detect(  # pyright: ignore
            image
        )
        pose_result = self.pose_landmarker.detect(  # pyright: ignore
            image
        )

        pose = False
        hands = False

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
                    wrist = np.array(
                        [
                            hand_landmarks[0].x,
                            hand_landmarks[0].y,
                            hand_landmarks[0].z,
                        ]
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

        else:
            if pose_result.pose_world_landmarks:
                pose = True
                poses = [
                    [pose.x, pose.y, pose.z]
                    for _, pose in enumerate(pose_result.pose_world_landmarks[0])
                ]
                results[0] = poses[0]  # nose
                results[1] = poses[12]  # right shoulder
                results[2] = poses[11]  # left shoulder
                results[3] = poses[14]  # right elbow
                results[4] = poses[13]  # left elbow
                results[5] = poses[16]  # right wrist
                results[6] = poses[15]  # left wrist

            if hand_result.hand_world_landmarks:
                hands = True
                for hand_landmarks, handedness in zip(
                    hand_result.hand_world_landmarks, hand_result.handedness
                ):
                    hand = handedness[0].category_name.lower()
                    wrist = np.array(
                        [
                            hand_landmarks[0].x,
                            hand_landmarks[0].y,
                            hand_landmarks[0].z,
                        ]
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

        return results, pose, hands

    def process_frame(
        self,
        frame: cv.typing.MatLike,
        timestamp: float,
        world_coords: bool = False,
    ) -> tuple[np.ndarray, bool, bool]:
        processed_frame = cv.flip(frame, 1)
        processed_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=processed_frame)

        results = np.empty((49, 3), dtype=np.float64)
        results[:] = np.nan

        hands = False
        pose = False

        int_timestamp = int(timestamp * 1000)
        if self.config.vision_mode == RunningMode.IMAGE:
            results, pose, hands = self.process_image(mp_image, world_coords)
        elif self.config.vision_mode == RunningMode.VIDEO:
            results, pose, hands = self.process_video(
                mp_image, int_timestamp, world_coords
            )
        elif self.config.vision_mode == RunningMode.LIVE_STREAM:
            self.process_live_stream(mp_image, int_timestamp)

        return_arr = np.array(results)

        # compensate for the wrist
        if hands:
            return_arr[7:28] += return_arr[5]
            return_arr[28:49] += return_arr[6]

        flat_result = results.flatten()

        all_data = np.hstack([timestamp, flat_result], dtype=np.float64)

        return all_data, hands, pose  # results_arr, empty_frame

    def process(self) -> None: ...

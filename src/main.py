# from time import time
from collections import OrderedDict
from config import ProcessingConfig
from live_processor import LiveProsessor

# from video_processor import VideoProcessor
from mediapipe.tasks.python.vision import RunningMode
import numpy as np


class MaxSizeDict(OrderedDict):
    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        super().__init__()

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


def main():
    frame_data: dict = MaxSizeDict(60)

    def face_callback(result, _image, timestamp_ms):
        # if timestamp_ms in frame_data:
        #     frame_data[timestamp_ms]["face_landmarks"] = result
        # else:
        #     frame_data[timestamp_ms] = {"face_landmarks": result}
        return

    def hand_callback(result, _image, timestamp_ms):
        if result.hand_landmarks:
            if timestamp_ms not in frame_data:
                frame_data[timestamp_ms] = [[-1, -1, -1]] * 49

            for hand_landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                hand = handedness[0].category_name.lower()
                if hand == "right":
                    frame_data[timestamp_ms][7:28] = [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in hand_landmarks
                    ]
                elif hand == "left":
                    frame_data[timestamp_ms][28:49] = [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in hand_landmarks
                    ]

    def pose_callback(result, _image, timestamp_ms):
        if result.pose_landmarks:
            if timestamp_ms not in frame_data:
                frame_data[timestamp_ms] = [[-1, -1, -1]] * 49

            poses = [
                [pose.x, pose.y, pose.z]
                for _, pose in enumerate(result.pose_landmarks[0])
            ]
            frame_data[timestamp_ms][0] = poses[0]
            frame_data[timestamp_ms][1] = poses[12]
            frame_data[timestamp_ms][2] = poses[11]
            frame_data[timestamp_ms][3] = poses[14]
            frame_data[timestamp_ms][4] = poses[13]
            frame_data[timestamp_ms][5] = poses[16]
            frame_data[timestamp_ms][6] = poses[15]

    config = ProcessingConfig(
        display_output=False,
        save_json=True,
        save_tfrecord=True,
        vision_mode=RunningMode.LIVE_STREAM,
        face_callback=face_callback,
        hand_callback=hand_callback,
        pose_callback=pose_callback,
    )

    processor = LiveProsessor(config)

    processor.process()

    list_data = []
    for key, item in frame_data.items():
        d = np.array([key, *np.array(item).flatten()])
        list_data.append(d)

    with open("test.csv", "w+") as f:
        headers = [
            "timestamp",
        ]
        labels = [
            "nose",
            "shoulder_r",
            "shoulder_l",
            "elbow_r",
            "elbow_r",
            "wrist_r",
            "wrist_l",
            *[f"hand_r_{i}" for i in range(21)],
            *[f"hand_l_{i}" for i in range(21)],
        ]
        coords = ["x", "y", "z"]

        for label in labels:
            for coord in coords:
                headers.append(f"{label}_{coord}")
        f.write(",".join(headers) + "\n")
        for line in list_data:
            s = [str(d) for d in line]
            f.write(",".join(s) + "\n")
    # with open("data.csv", "w+") as f:
    #     data = list(np.array(list(frame_data.items())).flatten())
    #     data_str = [",".join(d) for d in data]
    #     f.writelines(data_str)

    # Process a single video
    # for word in ["abort", "kaos", "melke", "sex", "skriver", "skyve", "vin"]:
    #     video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    #     processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

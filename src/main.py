# from time import time
from sys import maxsize
from config import ProcessingConfig
# from data_ensurer_pipeline import DataEnsurerPipeline

# from grid_subscriber import GridSubscriber
# from interpolation_subscriber import InterpolationSubscriber
# from grid_pipeline import GridPipeline
from collector_pipeline import CollectorPipeline
from interpolation_pipeline import InterpolationPipeline
from live_processor import LiveProsessor

# from video_processor import VideoProcessor
from mediapipe.tasks.python.vision import RunningMode

from logger_pipeline import LoggerPipeline
from movement_pipeline import MovementPipeline
from pairwise_pipeline import PairwisePipeline
from pipeline import PipelineManager
import numpy as np

from recorder_pipeline import RecorderPipeline
from video_processor import VideoProcessor


def main():
    pipeline = PipelineManager()

    def hand_callback(result, _, timestamp):
        if result.hand_landmarks:
            results = np.empty((49, 3))
            results[:] = np.nan
            for hand_landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
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
            pipeline.process(
                {"timestamp": timestamp, "hands": True, "results": results}
            )
        return

    def pose_callback(result, _, timestamp):
        if result.pose_landmarks:
            results = np.empty((49, 3))
            results[:] = np.nan
            poses = [
                [pose.x, pose.y, pose.z]
                for _, pose in enumerate(result.pose_landmarks[0])
            ]
            results[0] = poses[0]
            results[1] = poses[12]
            results[2] = poses[11]
            results[3] = poses[14]
            results[4] = poses[13]
            results[5] = poses[16]
            results[6] = poses[15]
            pipeline.process({"timestamp": timestamp, "pose": True, "results": results})
        return

    def face_callback(result, image, timestamp):
        return

    config = ProcessingConfig(
        display_output=False,
        save_json=True,
        save_tfrecord=True,
        vision_mode=RunningMode.IMAGE,
        pipeline=pipeline,
        hand_callback=hand_callback,
        pose_callback=pose_callback,
        face_callback=face_callback,
    )

    movement = MovementPipeline()
    pairwise = PairwisePipeline()
    collector = CollectorPipeline(max_size=None)
    # interpolator = InterpolationPipeline()
    recorder = RecorderPipeline(config)
    logger1 = LoggerPipeline()
    logger2 = LoggerPipeline()
    logger3 = LoggerPipeline()

    # collect the data
    pipeline.add_component(collector)

    # interpolate it
    # pipeline.add_component(interpolator)

    # pipeline.add_component(logger1)

    # calculate the movement
    pipeline.add_component(movement)
    # pipeline.add_component(logger2)

    # calculate distances
    pipeline.add_component(pairwise)
    # pipeline.add_component(logger3)
    #
    # # record

    # get distinct signs

    # predict the signs meaning

    # processor = LiveProsessor(config)
    videos = ["test.mp4"]

    processor = VideoProcessor(config, videos)

    processor.process()


if __name__ == "__main__":
    main()

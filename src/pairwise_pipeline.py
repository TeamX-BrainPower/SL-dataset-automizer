from typing import Any
from pipeline import PipelineComponent
import numpy as np


class PairwisePipeline(PipelineComponent):
    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return

        # get distance between
        # left and right hand
        # hands and nose
        # hands and shoulders

        # results[1]    # nose
        # results[2]    # right shoulder
        # results[3]    # left shoulder
        # results[4]    # right elbow
        # results[5]    # left elbow
        # results[6]    # right wrist
        # results[7]    # left wrist

        nose = data[:, 1:4]
        right_shoulder = data[:, 4:7]
        left_shoulder = data[:, 7:10]
        right_elbow = data[:, 10:13]
        left_elbow = data[:, 13:16]
        right_wrist = data[:, 16:19]
        left_wrist = data[:, 19:22]

        # general distances in pose
        distance_wrists = np.linalg.norm(right_wrist - left_wrist, axis=1)
        distance_elbows = np.linalg.norm(right_elbow - left_elbow, axis=1)
        distance_shoulders = np.linalg.norm(right_shoulder - left_shoulder, axis=1)

        # distances right side
        # we assume the distance between wrist elbow and elbow shoulder is constant
        distance_right_wrist_shoulder = np.linalg.norm(
            right_wrist - right_shoulder, axis=1
        )
        distance_right_wrist_nose = np.linalg.norm(right_wrist - nose, axis=1)
        distance_right_elbow_nose = np.linalg.norm(right_elbow - nose, axis=1)

        # distances left side
        distance_left_wrist_shoulder = np.linalg.norm(
            left_wrist - left_shoulder, axis=1
        )
        distance_left_wrist_nose = np.linalg.norm(left_wrist - nose, axis=1)
        distance_left_elbow_nose = np.linalg.norm(left_elbow - nose, axis=1)

        pose_distances = np.array(
            [
                distance_wrists,
                distance_elbows,
                distance_shoulders,
                distance_right_wrist_shoulder,
                distance_right_wrist_nose,
                distance_right_elbow_nose,
                distance_left_wrist_shoulder,
                distance_left_wrist_nose,
                distance_left_elbow_nose,
            ]
        ).T

        # print(data.shape, distance_wrists.shape, pose_distances.shape)
        # general distance in hands
        # 0 is wrist
        # 4 is thumb
        # 8 is index
        # 12 is middle
        # 16 is ring
        # 20 is little
        # results[8*3:29*3] = right hand
        # results[29*3:(29+21)*3] = left hand

        right_hand_wrist = data[:, 8 * 3 : 9 * 3]
        right_hand_thumb = data[:, 12 * 3 : 13 * 3]
        right_hand_index = data[:, 16 * 3 : 17 * 3]
        right_hand_middle = data[:, 20 * 3 : 21 * 3]
        right_hand_ring = data[:, 24 * 3 : 25 * 3]
        right_hand_little = data[:, 28 * 3 : 29 * 3]

        left_hand_wrist = data[:, 29 * 3 : 30 * 3]
        left_hand_thumb = data[:, 33 * 3 : 34 * 3]
        left_hand_index = data[:, 37 * 3 : 38 * 3]
        left_hand_middle = data[:, 41 * 3 : 42 * 3]
        left_hand_ring = data[:, 45 * 3 : 46 * 3]
        left_hand_little = data[:, 49 * 3 : 50 * 3]

        right_distance_thumb_wrist = np.linalg.norm(
            right_hand_wrist - right_hand_thumb, axis=1
        )
        right_distance_index_wrist = np.linalg.norm(
            right_hand_wrist - right_hand_index, axis=1
        )
        right_distance_middle_wrist = np.linalg.norm(
            right_hand_wrist - right_hand_middle, axis=1
        )
        right_distance_ring_wrist = np.linalg.norm(
            right_hand_wrist - right_hand_ring, axis=1
        )
        right_distance_little_wrist = np.linalg.norm(
            right_hand_wrist - right_hand_little, axis=1
        )

        right_distance_thumb_index = np.linalg.norm(
            right_hand_thumb - right_hand_index, axis=1
        )
        right_distance_index_middle = np.linalg.norm(
            right_hand_index - right_hand_middle, axis=1
        )
        right_distance_middle_ring = np.linalg.norm(
            right_hand_middle - right_hand_ring, axis=1
        )
        right_distance_ring_little = np.linalg.norm(
            right_hand_ring - right_hand_little, axis=1
        )

        left_distance_thumb_wrist = np.linalg.norm(
            left_hand_wrist - left_hand_thumb, axis=1
        )
        left_distance_index_wrist = np.linalg.norm(
            left_hand_wrist - left_hand_index, axis=1
        )
        left_distance_middle_wrist = np.linalg.norm(
            left_hand_wrist - left_hand_middle, axis=1
        )
        left_distance_ring_wrist = np.linalg.norm(
            left_hand_wrist - left_hand_ring, axis=1
        )
        left_distance_little_wrist = np.linalg.norm(
            left_hand_wrist - left_hand_little, axis=1
        )

        left_distance_thumb_index = np.linalg.norm(
            left_hand_thumb - left_hand_index, axis=1
        )
        left_distance_index_middle = np.linalg.norm(
            left_hand_index - left_hand_middle, axis=1
        )
        left_distance_middle_ring = np.linalg.norm(
            left_hand_middle - left_hand_ring, axis=1
        )
        left_distance_ring_little = np.linalg.norm(
            left_hand_ring - left_hand_little, axis=1
        )

        hand_distances = np.array(
            [
                right_distance_thumb_wrist,
                right_distance_index_wrist,
                right_distance_middle_wrist,
                right_distance_ring_wrist,
                right_distance_little_wrist,
                right_distance_thumb_index,
                right_distance_index_middle,
                right_distance_middle_ring,
                right_distance_ring_little,
                left_distance_thumb_wrist,
                left_distance_index_wrist,
                left_distance_middle_wrist,
                left_distance_ring_wrist,
                left_distance_little_wrist,
                left_distance_thumb_index,
                left_distance_index_middle,
                left_distance_middle_ring,
                left_distance_ring_little,
            ]
        ).T

        return np.hstack([data, pose_distances, hand_distances])

    def get_data(self) -> Any:
        return

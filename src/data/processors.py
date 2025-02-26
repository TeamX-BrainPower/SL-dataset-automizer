from abc import ABC, abstractmethod
import json
import tensorflow as tf

"""
Processors for handling frame data from MediaPipe solutions and saving it to either JSON or TFRecord format. 
"""


class DataProcessor(ABC):
    @abstractmethod
    def process_hands(self, frame_data):
        pass

    @abstractmethod
    def process_face(self, frame_data):
        pass

    @abstractmethod
    def process_gesture(self, frame_data):
        pass

    @abstractmethod
    def process_pose(self, frame_data):
        pass

    @abstractmethod
    def save(self, filename):
        pass


class JSONProcessor(DataProcessor):
    def __init__(self, word, total_frames, frame_rate):
        self.data = {
            "word": word,
            "total_frame_count": total_frames,
            "video_frame_rate": frame_rate,
            "frameData": []
        }

    def process_hands(self, hand_result):
        if hand_result and hand_result.hand_landmarks:
            self.frame_info["hands"] = []
            for landmarks, handedness in zip(
                    hand_result.hand_landmarks,
                    hand_result.handedness
            ):
                hand_info = {
                    "handedness": handedness[0].category_name,
                    "landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in landmarks
                    ]
                }
                self.frame_info["hands"].append(hand_info)

    
    def process_face(self, face_result):
        if face_result and face_result.face_landmarks:
            self.frame_info["face"] = []
            face_info = {
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for face_landmarks in face_result.face_landmarks
                    for lm in face_landmarks
                ]
            }
            self.frame_info["face"] = face_info

    def process_gesture(self, gesture_result):
        if gesture_result and gesture_result.gestures:
            self.frame_info["gestures"] = []
            for gesture in gesture_result.gestures:
                gesture_info = {
                    "categoryName": gesture[0].category_name,
                    "score": float(gesture[0].score),
                    "index": int(gesture[0].index)
                }
                self.frame_info["gestures"].append(gesture_info)

    def process_pose(self, pose_result):
        if pose_result and pose_result.pose_landmarks:
            self.frame_info["pose"] = []
            for pose_landmarks in pose_result.pose_landmarks:
                pose_info = {
                    "landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in pose_landmarks
                    ]
                }
                self.frame_info["pose"].append(pose_info)

    def process_frame(self, frame_num, hand_result, face_result, gesture_result, pose_result):
        self.frame_info = {"frame": frame_num}
        if hand_result and hand_result.hand_landmarks:
            self.process_hands(hand_result)
        if face_result and face_result.face_landmarks:
            self.process_face(face_result)
        if gesture_result and gesture_result.gestures:
            self.process_gesture(gesture_result)
        if pose_result and pose_result.pose_landmarks:
            self.process_pose(pose_result)
        
        self.data["frameData"].append(self.frame_info)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)


class TFRecordProcessor(DataProcessor):
    def __init__(self, word):
        self.word = word
        self.features = []

    def process_hands(self, hand_result, frame_num):
        if hand_result and hand_result.hand_landmarks:
            # Convert landmarks to flat list for TFRecord
            landmarks_flat = []
            handedness = []
            for landmarks, hand in zip(hand_result.hand_landmarks, hand_result.handedness):
                handedness.append(hand[0].category_name.encode())
                for lm in landmarks:
                    landmarks_flat.extend([lm.x, lm.y, lm.z])

            feature = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'hand_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=landmarks_flat)),
                'handedness': tf.train.Feature(bytes_list=tf.train.BytesList(value=handedness))
            }
            self.features.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    def process_face(self, face_result, frame_num):
        if face_result and face_result.face_landmarks:
            # Convert face landmarks to flat list
            face_landmarks_flat = []
            for face_landmarks in face_result.face_landmarks:
                for lm in face_landmarks:
                    face_landmarks_flat.extend([lm.x, lm.y, lm.z])

            feature = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'face_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=face_landmarks_flat))
            }
            self.features.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    def process_gesture(self, gesture_result, frame_num):
        if gesture_result and gesture_result.gestures:
            # Convert gesture data
            categories = []
            scores = []
            indices = []
            for gesture in gesture_result.gestures:
                categories.append(gesture[0].category_name.encode())
                scores.append(float(gesture[0].score))
                indices.append(int(gesture[0].index))

            feature = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'gesture_categories': tf.train.Feature(bytes_list=tf.train.BytesList(value=categories)),
                'gesture_scores': tf.train.Feature(float_list=tf.train.FloatList(value=scores)),
                'gesture_indices': tf.train.Feature(int64_list=tf.train.Int64List(value=indices))
            }
            self.features.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    def process_pose(self, pose_result, frame_num):
        if pose_result and pose_result.pose_landmarks:
            # Convert pose landmarks to flat list
            pose_landmarks_flat = []
            for landmarks in pose_result.pose_landmarks:
                for lm in landmarks:
                    pose_landmarks_flat.extend([lm.x, lm.y, lm.z])

            feature = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'pose_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=pose_landmarks_flat))
            }
            self.features.append(tf.train.Example(features=tf.train.Features(feature=feature)))


    def save(self, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            for example in self.features:
                writer.write(example.SerializeToString())
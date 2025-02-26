from abc import ABC, abstractmethod
import json



class DataProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame_data):
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

    def process_frame(self, hand_result, frame_num):
        frame_info = {"frame": frame_num, "hands": []}
        if hand_result and hand_result.hand_landmarks:
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
                frame_info["hands"].append(hand_info)
        self.data["frameData"].append(frame_info)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)


class TFRecordProcessor(DataProcessor):
    import tensorflow as tf
    def __init__(self, word):
        self.word = word
        self.features = []

    def process_frame(self, hand_result, frame_num):
        if hand_result and hand_result.hand_landmarks:
            # Convert landmarks to flat list for TFRecord
            landmarks_flat = []
            for landmarks in hand_result.hand_landmarks:
                for lm in landmarks:
                    landmarks_flat.extend([lm.x, lm.y, lm.z])

            # Create feature
            feature = {
                'word': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[self.word.encode()])
                ),
                'frame': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[frame_num])
                ),
                'landmarks': tf.train.Feature(
                    float_list=tf.train.FloatList(value=landmarks_flat)
                )
            }
            self.features.append(tf.train.Example(
                features=tf.train.Features(feature=feature)
            ))

    def save(self, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            for example in self.features:
                writer.write(example.SerializeToString())

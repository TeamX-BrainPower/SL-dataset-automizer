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

    def calculate_face_center(self, pose_result):
        """Calculate the face center (nose point) from pose landmarks"""
        if pose_result and pose_result.pose_world_landmarks:
            for landmarks in pose_result.pose_world_landmarks:
                # The nose is the 0th landmark in MediaPipe pose
                nose = landmarks[0]
                return {"x": nose.x, "y": nose.y, "z": nose.z}
        return None

    def get_wrist_positions(self, pose_result):
        """Get left and right wrist positions from pose landmarks"""
        left_wrist = None
        right_wrist = None
        
        if pose_result and pose_result.pose_world_landmarks:
            for landmarks in pose_result.pose_world_landmarks:
                # Left wrist is index 15, right wrist is index 16 in MediaPipe pose
                if len(landmarks) > 16:
                    left_wrist = {"x": landmarks[15].x, "y": landmarks[15].y, "z": landmarks[15].z}
                    right_wrist = {"x": landmarks[16].x, "y": landmarks[16].y, "z": landmarks[16].z}
        
        return left_wrist, right_wrist

    def process_hands(self, hand_result, face_center=None, left_wrist=None, right_wrist=None):
        if hand_result and hand_result.hand_world_landmarks:
            self.frame_info["hands"] = []
            
            for landmarks, handedness in zip(
                    hand_result.hand_world_landmarks,
                    hand_result.handedness
            ):
                hand_key = handedness[0].category_name
                
                # Get corresponding wrist position from pose
                wrist_pos = left_wrist if hand_key == "Left" else right_wrist
                
                # Initialize list for landmarks
                processed_landmarks = []
                
                # Calculate offset between hand wrist and pose wrist if both are available
                offset_x = 0
                offset_y = 0
                offset_z = 0
                
                if wrist_pos and face_center:
                    # MediaPipe hand wrist is at index 0
                    hand_wrist_x = landmarks[0].x
                    hand_wrist_y = landmarks[0].y
                    hand_wrist_z = landmarks[0].z
                    
                    # Calculate offset to align hand wrist with pose wrist
                    offset_x = wrist_pos["x"] - hand_wrist_x
                    offset_y = wrist_pos["y"] - hand_wrist_y
                    offset_z = wrist_pos["z"] - hand_wrist_z
                
                # Process all landmarks with centering around face and wrist alignment
                for lm in landmarks:
                    x_centered = lm.x
                    y_centered = lm.y
                    z_centered = lm.z
                    
                    # Apply offset for wrist alignment
                    x_centered += offset_x
                    y_centered += offset_y
                    z_centered += offset_z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    processed_landmarks.append({
                        "x": x_centered,
                        "y": y_centered,
                        "z": z_centered
                    })
                
                hand_info = {
                    "handedness": hand_key,
                    "landmarks": processed_landmarks
                }
                self.frame_info["hands"].append(hand_info)

    def process_face(self, face_result, face_center=None):
        if face_result and face_result.face_landmarks:
            self.frame_info["face"] = []
            
            processed_landmarks = []
            
            for face_landmarks in face_result.face_landmarks:
                for lm in face_landmarks:
                    x_centered = lm.x
                    y_centered = lm.y
                    z_centered = lm.z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    processed_landmarks.append({
                        "x": x_centered,
                        "y": y_centered,
                        "z": z_centered
                    })
            
            face_info = {
                "landmarks": processed_landmarks
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

    def process_pose(self, pose_result, face_center=None):
        if pose_result and pose_result.pose_world_landmarks:
            self.frame_info["pose"] = []
            
            for pose_landmarks in pose_result.pose_world_landmarks:
                processed_landmarks = []
                
                for lm in pose_landmarks:
                    x_centered = lm.x
                    y_centered = lm.y
                    z_centered = lm.z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    processed_landmarks.append({
                        "x": x_centered,
                        "y": y_centered,
                        "z": z_centered
                    })
                
                pose_info = {
                    "landmarks": processed_landmarks
                }
                self.frame_info["pose"].append(pose_info)

    def process_frame(self, frame_num, hand_result, face_result, gesture_result, pose_result):
        self.frame_info = {"frame": frame_num}
        
        # Calculate face center (nose position) if pose data is available
        face_center = self.calculate_face_center(pose_result)
        
        # Get wrist positions from pose if available
        left_wrist, right_wrist = self.get_wrist_positions(pose_result)
        
        # Process each type of data with centering around face and proper alignment
        if pose_result and pose_result.pose_landmarks:
            self.process_pose(pose_result, face_center)
        
        if hand_result and hand_result.hand_landmarks:
            self.process_hands(hand_result, face_center, left_wrist, right_wrist)
            
        if face_result and face_result.face_landmarks:
            self.process_face(face_result, face_center)
            
        if gesture_result and gesture_result.gestures:
            self.process_gesture(gesture_result)
        
        self.data["frameData"].append(self.frame_info)

        self.data["total_frame_count"] = frame_num + 1

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)


class TFRecordProcessor(DataProcessor):
    def __init__(self, word):
        self.word = word
        self.features = []
        self.current_frame_features = {}

    def calculate_face_center(self, pose_result):
        """Calculate the face center (nose point) from pose landmarks"""
        if pose_result and pose_result.pose_world_landmarks:
            for landmarks in pose_result.pose_world_landmarks:
                # The nose is the 0th landmark in MediaPipe pose
                nose = landmarks[0]
                return {"x": nose.x, "y": nose.y, "z": nose.z}
        return None

    def get_wrist_positions(self, pose_result):
        """Get left and right wrist positions from pose landmarks"""
        left_wrist = None
        right_wrist = None
        
        if pose_result and pose_result.pose_world_landmarks:
            for landmarks in pose_result.pose_world_landmarks:
                # Left wrist is index 15, right wrist is index 16 in MediaPipe pose
                if len(landmarks) > 16:
                    left_wrist = {"x": landmarks[15].x, "y": landmarks[15].y, "z": landmarks[15].z}
                    right_wrist = {"x": landmarks[16].x, "y": landmarks[16].y, "z": landmarks[16].z}
        
        return left_wrist, right_wrist

    def process_hands(self, hand_result, frame_num, face_center=None, left_wrist=None, right_wrist=None):
        if hand_result and hand_result.hand_landmarks:
            # Convert landmarks to flat list for TFRecord with normalization
            landmarks_flat = []
            handedness = []
            
            for landmarks, hand in zip(hand_result.hand_landmarks, hand_result.handedness):
                hand_key = hand[0].category_name
                handedness.append(hand_key.encode())
                
                # Get corresponding wrist position from pose
                wrist_pos = left_wrist if hand_key == "Left" else right_wrist
                
                # Calculate offset between hand wrist and pose wrist if both are available
                offset_x = 0
                offset_y = 0
                offset_z = 0
                
                if wrist_pos and face_center:
                    # MediaPipe hand wrist is at index 0
                    hand_wrist_x = landmarks[0].x
                    hand_wrist_y = landmarks[0].y
                    hand_wrist_z = landmarks[0].z
                    
                    # Calculate offset to align hand wrist with pose wrist
                    offset_x = wrist_pos["x"] - hand_wrist_x
                    offset_y = wrist_pos["y"] - hand_wrist_y
                    offset_z = wrist_pos["z"] - hand_wrist_z
                
                # Process all landmarks with normalization
                for lm in landmarks:
                    x_centered = lm.x + offset_x
                    y_centered = lm.y + offset_y
                    z_centered = lm.z + offset_z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    landmarks_flat.extend([x_centered, y_centered, z_centered])

            self.current_frame_features = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'hand_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=landmarks_flat)),
                'handedness': tf.train.Feature(bytes_list=tf.train.BytesList(value=handedness))
            }
            
            self.features.append(tf.train.Example(
                features=tf.train.Features(feature=self.current_frame_features)
            ))

    def process_face(self, face_result, frame_num, face_center=None):
        if face_result and face_result.face_landmarks:
            # Convert face landmarks to flat list with normalization
            face_landmarks_flat = []
            
            for face_landmarks in face_result.face_landmarks:
                for lm in face_landmarks:
                    x_centered = lm.x
                    y_centered = lm.y
                    z_centered = lm.z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    face_landmarks_flat.extend([x_centered, y_centered, z_centered])

            self.current_frame_features = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'face_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=face_landmarks_flat))
            }
            
            self.features.append(tf.train.Example(
                features=tf.train.Features(feature=self.current_frame_features)
            ))

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

            self.current_frame_features = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'gesture_categories': tf.train.Feature(bytes_list=tf.train.BytesList(value=categories)),
                'gesture_scores': tf.train.Feature(float_list=tf.train.FloatList(value=scores)),
                'gesture_indices': tf.train.Feature(int64_list=tf.train.Int64List(value=indices))
            }
            
            self.features.append(tf.train.Example(
                features=tf.train.Features(feature=self.current_frame_features)
            ))

    def process_pose(self, pose_result, frame_num, face_center=None):
        if pose_result and pose_result.pose_landmarks:
            # Convert pose landmarks to flat list with normalization
            pose_landmarks_flat = []
            
            for landmarks in pose_result.pose_landmarks:
                for lm in landmarks:
                    x_centered = lm.x
                    y_centered = lm.y
                    z_centered = lm.z
                    
                    # Center around face if face center is available
                    if face_center:
                        x_centered -= face_center["x"]
                        y_centered -= face_center["y"]
                        z_centered -= face_center["z"]
                    
                    pose_landmarks_flat.extend([x_centered, y_centered, z_centered])

            self.current_frame_features = {
                'word': tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.word.encode()])),
                'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_num])),
                'pose_landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=pose_landmarks_flat))
            }
            
            self.features.append(tf.train.Example(
                features=tf.train.Features(feature=self.current_frame_features)
            ))
            
    def process_frame(self, frame_num, hand_result, face_result, gesture_result, pose_result):
        """Process all data for a single frame with normalization and alignment"""
        # Calculate face center (nose position) if pose data is available
        face_center = self.calculate_face_center(pose_result)
        
        # Get wrist positions from pose if available
        left_wrist, right_wrist = self.get_wrist_positions(pose_result)
        
        # Process each type of data with centering around face and proper alignment
        if pose_result and pose_result.pose_landmarks:
            self.process_pose(pose_result, frame_num, face_center)
        
        if hand_result and hand_result.hand_landmarks:
            self.process_hands(hand_result, frame_num, face_center, left_wrist, right_wrist)
            
        # if face_result and face_result.face_landmarks:
        #     self.process_face(face_result, frame_num, face_center)
            
        # if gesture_result and gesture_result.gestures:
        #     self.process_gesture(gesture_result, frame_num)

    def save(self, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            for example in self.features:
                writer.write(example.SerializeToString())
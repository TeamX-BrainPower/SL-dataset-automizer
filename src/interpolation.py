from scipy.interpolate import interp1d
import numpy as np
import json
from visualization.hand_trajectory import display_dynamic, process_landmarks


class DataReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.json_data = self.load_json()
        self.orig_numpy_data = self.json_to_numpy()
        self.data = self.zero_shift_array()
        # data[pose/left/right][frame][landmark][x/y/z]
        
    def load_json(self):
        with open(f'data/raw/{self.file_name}.json') as f:
            data = json.load(f)
        return data
    
    def json_to_numpy(self):
        frames = self.json_data.get("frameData")
        
        pose_data = []
        left_hand_data = []
        right_hand_data = []
        
        for frame in frames:
            pose_array = np.full((33, 3), np.nan)
            left_hand_array = np.full((21, 3), np.nan)
            right_hand_array = np.full((21, 3), np.nan)
            
            if "pose" in frame and frame["pose"]:
                pose_landmarks = frame["pose"][0]["landmarks"]
                pose_array = np.array([[lm["x"], lm["y"], lm["z"]] for lm in pose_landmarks])
            
            for hand in frame.get("hands", []):
                hand_landmarks = hand["landmarks"]
                if hand["handedness"] == "Left":
                    left_hand_array = np.array([[lm["x"], lm["y"], lm["z"]] for lm in hand_landmarks])
                else:
                    right_hand_array = np.array([[lm["x"], lm["y"], lm["z"]] for lm in hand_landmarks])
            
            pose_data.append(pose_array)
            left_hand_data.append(left_hand_array)
            right_hand_data.append(right_hand_array)
        
        return np.array([pose_data, left_hand_data, right_hand_data], dtype=object)
    
    def zero_shift_array(self):
        [pose_data, left_hand_data, right_hand_data] = self.orig_numpy_data
        first = 0
        for i in range(len(pose_data)):
            if not np.isnan(left_hand_data[i][0][0]) or not np.isnan(right_hand_data[i][0][0]):
                first = i
                break
        
        pose_data = pose_data[first:]
        left_hand_data = left_hand_data[first:]
        right_hand_data = right_hand_data[first:]
        
        return np.array([pose_data, left_hand_data, right_hand_data], dtype=object)



class Interpolator:
    def __init__(self, data, n_frames=60):
        self.data = data
        # data[pose/left/right][frame][landmark][x/y/z]
        self.length = len(data[0])
        self.n_frames = n_frames
        self.interp_data = self.interpolate()
        
    def interpolate(self):
        interp_data = []
        for array in self.data: # pose/left/right
            new_array = np.full((self.n_frames, len(array[0]), 3), np.nan)
            for j in range(len(array[0])): # no. of landmarks
                for k in range(3): # x/y/z
                    x = np.arange(self.length)
                    y = np.array([array[i][j][k] for i in range(self.length)]) # for each frame i, the value of coord k for landmark j
                    
                    valid_indices = np.where(~np.isnan(y))[0]
                    valid_values = y[valid_indices]

                    interpolator = interp1d(valid_indices, valid_values, fill_value='extrapolate')
                    y_interp = interpolator(x)
                    
                    x_new = np.linspace(0, self.length - 1, self.n_frames)
                    resampler = interp1d(x, y_interp)
                    
                    new_array[:, j, k] = resampler(x_new)
                        
            interp_data.append(new_array)
        return interp_data


class DataWriter:
    def __init__(self, file_name, numpy_data):
        self.file_name = file_name
        self.numpy_data = numpy_data
        self.json_data = self.numpy_to_json()
        
    def write_json(self):
        json_data = json.dumps(self.json_data, indent=4)
        with open(f'data/raw/{self.file_name}_interp.json', 'w') as f:
            f.write(json_data)
            
    def numpy_to_json(self):
        [pose_data, left_hand_data, right_hand_data] = self.numpy_data
        json_data = {"word": self.file_name,
                     "total_frame_count": len(pose_data),
                     "video_frame_rate": len(pose_data),
                     "frameData": []
                     }
        
        for i in range(len(pose_data)):
            frame = {"frame": i, 
                     "pose": [],
                     "hands": []
                     }
            
            pose_landmarks = [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in pose_data[i]]
            frame["pose"].append({"landmarks": pose_landmarks})

            left_landmarks = [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in left_hand_data[i]]
            frame["hands"].append({"handedness": "Left", "landmarks": left_landmarks})

            right_landmarks = [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in right_hand_data[i]]
            frame["hands"].append({"handedness": "Right", "landmarks": right_landmarks})
            
            json_data["frameData"].append(frame)
        
        return json_data


class HandTrajectoryProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_reader = DataReader(file_name)
        self.interpolator = Interpolator(self.data_reader.data)
        self.writer = DataWriter(file_name, self.interpolator.interp_data)

    def process(self):
        interpolated_data = self.interpolator.interpolate()
        # display_dynamic(interpolated_data)
        self.writer.write_json()
        


if __name__ == "__main__":
    processor = HandTrajectoryProcessor('skilsmisse')
    processor.process()

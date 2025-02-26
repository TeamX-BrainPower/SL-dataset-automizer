from scipy.interpolate import interp1d
import numpy as np
import json

from visualization.hand_trajectory import display_dynamic

def main():
    data = load_json('skilsmisse')
    interp_data = interpolate(data)
    print(interp_data)
    # display_dynamic(interp_data)

# Load and return file_name.json
def load_json(file_name):
    with open(f'data/raw/{file_name}.json') as f:
        data = json.load(f)
    return data
    
# Process landmarks for display functions
def process_landmarks(data):
    landmark_trajectories = {
        "Left": {},
        "Right": {}
    }
    length = data["total_frame_count"]
    for frame in data["frameData"]:
        for hand in frame["hands"]:
            hand_marks = hand["landmarks"]
            hand_key = hand["handedness"]
            for i, landmark in enumerate(hand_marks):
                if i not in landmark_trajectories[hand_key]:
                    landmark_trajectories[hand_key][i] = {"x": [None] * length , "y": [None] * length, "z": [None] * length}
                landmark_trajectories[hand_key][i]["x"][frame["frame"]] = landmark["x"]
                landmark_trajectories[hand_key][i]["y"][frame["frame"]] = landmark["y"]
                landmark_trajectories[hand_key][i]["z"][frame["frame"]] = landmark["z"]
                
    return landmark_trajectories  

def find_first_nonempty_frame(data):
    first = 0
    for frame in data["frameData"]:
        if frame["hands"]:
            first = frame["frame"]
            break
        
    return first

def interpolate(data, n_frames=30):
    first = find_first_nonempty_frame(data)
    length = data["total_frame_count"] - first
    traj = process_landmarks(data)
    shifted_traj = {
        "Left": {},
        "Right": {}
    }
    
    for hand_key in ["Left", "Right"]:
        for t in range(0, length):
            for i in range(0,21):
                if traj[hand_key]:
                    if i not in shifted_traj[hand_key]:
                        shifted_traj[hand_key][i] = {"x": [None] * length , "y": [None] * length, "z": [None] * length}
                    shifted_traj[hand_key][i]["x"][t] = traj[hand_key][i]["x"][t + first]
                    shifted_traj[hand_key][i]["y"][t] = traj[hand_key][i]["y"][t + first]
                    shifted_traj[hand_key][i]["z"][t] = traj[hand_key][i]["z"][t + first]
    
    t_scaled_traj = {
        "Left": {},
        "Right": {}
    }
    
    for hand_key in ("Left", "Right"):
        for i in range(0, 21):
            if shifted_traj[hand_key]:
                for coord in ["x", "y", "z"]:
                    x = np.arange(length)
                    y = np.array(shifted_traj[hand_key][i][coord], dtype=float)
                    valid_indices = np.where(~np.isnan(y))[0]
                    valid_values = y[valid_indices]
                    
                    interpolator = interp1d(valid_indices, valid_values, fill_value='extrapolate')
                    y_interp = interpolator(x)
                    
                    x_new = np.linspace(0, length - 1, 30)
                    resampler = interp1d(x, y_interp)
                    if i not in t_scaled_traj[hand_key]:
                        t_scaled_traj[hand_key][i] = {}
                    t_scaled_traj[hand_key][i][coord] = resampler(x_new)
        
    return t_scaled_traj


if __name__ == "__main__":
    main()
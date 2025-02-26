from scipy.interpolate import interp1d
import numpy as np
import json
from visualization.hand_trajectory import display_dynamic, process_landmarks


class DataLoader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = self.load_json()

    def load_json(self):
        with open(f'data/raw/{self.file_name}.json') as f:
            data = json.load(f)
        return data


class Interpolator:
    def __init__(self, data, n_frames=30):
        self.data = data
        self.n_frames = n_frames
        self.first_frame = self.find_first_nonempty_frame()
        self.length = self.data["total_frame_count"] - self.first_frame
        self.traj = process_landmarks(self.data)
        self.shifted_traj = {
            "Left": {},
            "Right": {}
        }
        self.t_scaled_traj = {
            "Left": {},
            "Right": {}
        }

    def find_first_nonempty_frame(self):
        first = 0
        for frame in self.data["frameData"]:
            if frame["hands"]:
                first = frame["frame"]
                break
        return first

    def shift_trajectory(self):
        for hand_key in ["Left", "Right"]:
            for t in range(0, self.length):
                for i in range(0, 21):
                    if self.traj[hand_key]:
                        if i not in self.shifted_traj[hand_key]:
                            self.shifted_traj[hand_key][i] = {"x": [None] * self.length, 
                                                              "y": [None] * self.length, 
                                                              "z": [None] * self.length}
                        self.shifted_traj[hand_key][i]["x"][t] = self.traj[hand_key][i]["x"][t + self.first_frame]
                        self.shifted_traj[hand_key][i]["y"][t] = self.traj[hand_key][i]["y"][t + self.first_frame]
                        self.shifted_traj[hand_key][i]["z"][t] = self.traj[hand_key][i]["z"][t + self.first_frame]

    def interpolate(self):
        self.shift_trajectory()

        for hand_key in ("Left", "Right"):
            for i in range(0, 21):
                if self.shifted_traj[hand_key]:
                    for coord in ["x", "y", "z"]:
                        x = np.arange(self.length)
                        y = np.array(self.shifted_traj[hand_key][i][coord], dtype=float)
                        valid_indices = np.where(~np.isnan(y))[0]
                        valid_values = y[valid_indices]

                        interpolator = interp1d(valid_indices, valid_values, fill_value='extrapolate')
                        y_interp = interpolator(x)

                        x_new = np.linspace(0, self.length - 1, self.n_frames)
                        resampler = interp1d(x, y_interp)
                        if i not in self.t_scaled_traj[hand_key]:
                            self.t_scaled_traj[hand_key][i] = {}
                        self.t_scaled_traj[hand_key][i][coord] = resampler(x_new)

        return self.t_scaled_traj


class HandTrajectoryProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_loader = DataLoader(file_name)
        self.interpolator = Interpolator(self.data_loader.data)

    def process(self):
        interpolated_data = self.interpolator.interpolate()
        display_dynamic(interpolated_data)


if __name__ == "__main__":
    processor = HandTrajectoryProcessor('kaos')
    processor.process()

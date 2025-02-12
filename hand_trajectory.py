import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define colors for different fingers
finger_colors = {
    "wrist": "black",
    "thumb": "red",
    "index": "blue",
    "middle": "green",
    "ring": "purple",
    "pinky": "orange"
}

# Assign colors based on landmark index
landmark_colors = {
    0: finger_colors["wrist"],  # Wrist
    **{i: finger_colors["thumb"] for i in range(1, 5)},
    **{i: finger_colors["index"] for i in range(5, 9)},
    **{i: finger_colors["middle"] for i in range(9, 13)},
    **{i: finger_colors["ring"] for i in range(13, 17)},
    **{i: finger_colors["pinky"] for i in range(17, 21)}
}

# Finger connections based on landmark indices
finger_chains = [
    [1, 2, 3, 4],  # Thumb
    [5, 6, 7, 8],  # Index
    [9, 10, 11, 12],  # Middle
    [13, 14, 15, 16],  # Ring
    [17, 18, 19, 20],  # Pinky
]

# Wrist connection lines
wrist_connections = [1, 5, 9, 13, 17]

# Base line for fingers
base_line = [5, 9 ,13, 17]



# Main func
def main():
    data = load_json('abort')
    trajectories = process_landmarks(data)
    display_dynamic(trajectories)
    

# Load and return file_name.json
def load_json(file_name):
    with open(f'parsed-output/{file_name}.json') as f:
        data = json.load(f)
    return data


# Process landmarks for display functions
def process_landmarks(data):
    landmark_trajectories = {}
    
    for frame in data["frameData"]:
        for hand in frame["hands"]:
            hand_marks = hand["landmarks"]
            for i, landmark in enumerate(hand_marks):
                if i not in landmark_trajectories:
                    landmark_trajectories[i] = {"x": [], "y": [], "z": [], "frame": []}
                landmark_trajectories[i]["x"].append(landmark["x"])
                landmark_trajectories[i]["y"].append(landmark["y"])
                landmark_trajectories[i]["z"].append(landmark["z"])
                landmark_trajectories[i]["frame"].append(frame["frame"])
                
    return landmark_trajectories


# Displays dynamic 3d trajectory from data json via load_json()
def display_dynamic(landmark_trajectories, interval=50):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute min/max values for better zoom control
    x_min = min(min(traj["x"]) for traj in landmark_trajectories.values())
    x_max = max(max(traj["x"]) for traj in landmark_trajectories.values())
    y_min = min(min(traj["y"]) for traj in landmark_trajectories.values())
    y_max = max(max(traj["y"]) for traj in landmark_trajectories.values())
    z_min = min(min(traj["z"]) for traj in landmark_trajectories.values())
    z_max = max(max(traj["z"]) for traj in landmark_trajectories.values())

    margin = 0.1  # 10% margin
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    # Initialize line objects for each landmark
    lines = {i: ax.plot([], [], [], color=landmark_colors[i], linewidth=1)[0] for i in landmark_trajectories}

    # Initialize line objects for finger chains
    finger_lines = {tuple(chain): ax.plot([], [], [], color=landmark_colors[chain[0]], linewidth=2)[0] for chain in finger_chains}

    # Initialize lines for wrist connections
    wrist_lines = {i: ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0] for i in wrist_connections}

    # Initialize line for the additional base connection (5 → 9 → 13 → 17)
    base_line_plot = ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0]

    # Scatter points for the start of each trajectory
    start_points = {i: ax.scatter([], [], [], color=landmark_colors[i], s=50, zorder=5) for i in landmark_trajectories}

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Hand Landmark 3D Animated Trajectories")
    
    # Set the camera angle to resemble the video perspective
    ax.view_init(elev=90, azim=90)  # Set elevation and azimuthal angles here
    ax.dist = 8  # Set distance from the viewer to the plot

    max_frames = max(len(traj["frame"]) for traj in landmark_trajectories.values())

    def update(frame):
        """Update function for the animation"""
        # for i, trajectory in landmark_trajectories.items():
        #     x_data = trajectory["x"][:frame]
        #     y_data = trajectory["y"][:frame]
        #     z_data = trajectory["z"][:frame]

        #     # Update trajectory lines
        #     lines[i].set_data(x_data, y_data)
        #     lines[i].set_3d_properties(z_data)

        #     # Keep start points fixed
        #     if frame == 1:
        #         start_points[i]._offsets3d = (np.array([x_data[0]]), np.array([y_data[0]]), np.array([z_data[0]]))

        # Update finger chains
        for chain in finger_chains:
            x_chain = [landmark_trajectories[i]["x"][frame - 1] for i in chain]
            y_chain = [landmark_trajectories[i]["y"][frame - 1] for i in chain]
            z_chain = [landmark_trajectories[i]["z"][frame - 1] for i in chain]
            finger_lines[tuple(chain)].set_data(x_chain, y_chain)
            finger_lines[tuple(chain)].set_3d_properties(z_chain)

        # Update wrist connections
        for i in wrist_connections:
            x_wrist = [landmark_trajectories[0]["x"][frame - 1], landmark_trajectories[i]["x"][frame - 1]]
            y_wrist = [landmark_trajectories[0]["y"][frame - 1], landmark_trajectories[i]["y"][frame - 1]]
            z_wrist = [landmark_trajectories[0]["z"][frame - 1], landmark_trajectories[i]["z"][frame - 1]]
            wrist_lines[i].set_data(x_wrist, y_wrist)
            wrist_lines[i].set_3d_properties(z_wrist)

        # Update base connection line (5 → 9 → 13 → 17)
        x_base = [landmark_trajectories[i]["x"][frame - 1] for i in base_line]
        y_base = [landmark_trajectories[i]["y"][frame - 1] for i in base_line]
        z_base = [landmark_trajectories[i]["z"][frame - 1] for i in base_line]
        base_line_plot.set_data(x_base, y_base)
        base_line_plot.set_3d_properties(z_base)

        return list(lines.values()) + list(finger_lines.values()) + list(wrist_lines.values()) + list(start_points.values()) + [base_line_plot]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=True)

    plt.show()
    

if __name__ == "__main__":
    main()
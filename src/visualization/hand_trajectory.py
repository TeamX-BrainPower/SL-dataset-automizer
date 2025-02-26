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
base_line = [5, 9, 13, 17]


# Main func
def main():
    data = load_json('melke')
    landmark_trajectories = process_landmarks(data)
    display_dynamic(landmark_trajectories)
    

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


# Displays dynamic 3d trajectory
def display_dynamic(landmark_trajectories):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0

    # Compute min/max values for better zoom control based on (left) hand landmarks
    if landmark_trajectories["Left"].values():
        x_min = min(
            min(x for x in traj["x"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["x"])
        )
        x_max = max(
            max(x for x in traj["x"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["x"])
        )
        y_min = min(
            min(x for x in traj["y"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["y"])
        )
        y_max = max(
            max(x for x in traj["y"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["y"])
        )
        z_min = min(
            min(x for x in traj["z"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["z"])
        )
        z_max = max(
            max(x for x in traj["z"] if x is not None)
            for traj in landmark_trajectories["Left"].values()
            if any(x is not None for x in traj["z"])
        )

    # Consider the right hand as well
    if landmark_trajectories["Right"].values():
        x_min = min(x_min,
            *(min(x for x in traj["x"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["x"]))
        )
        x_max = max(x_max,
            *(max(x for x in traj["x"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["x"]))
        )
        y_min = min(y_min,
            *(min(x for x in traj["y"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["y"]))
        )
        y_max = max(y_max,
            *(max(x for x in traj["y"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["y"]))
        )
        z_min = min(z_min,
            *(min(x for x in traj["z"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["z"]))
        )
        z_max = max(z_max,
            *(max(x for x in traj["z"] if x is not None)
            for traj in landmark_trajectories["Right"].values()
            if any(x is not None for x in traj["z"]))
        )

    margin = 0.1  # 10% margin
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    # # Initialize line objects for each landmark (left and right hands)
    lines_left = {i: ax.plot([], [], [], color=landmark_colors[i], linewidth=1)[0] for i in landmark_trajectories["Left"]}
    lines_right = {i: ax.plot([], [], [], color=landmark_colors[i], linewidth=1)[0] for i in landmark_trajectories["Right"]}

    # Initialize line objects for finger chains (left and right hands)
    finger_lines_left = {tuple(chain): ax.plot([], [], [], color=landmark_colors[chain[0]], linewidth=2)[0] for chain in finger_chains}
    finger_lines_right = {tuple(chain): ax.plot([], [], [], color=landmark_colors[chain[0]], linewidth=2)[0] for chain in finger_chains}

    # Initialize lines for wrist connections
    wrist_lines_left = {i: ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0] for i in wrist_connections}
    wrist_lines_right = {i: ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0] for i in wrist_connections}

    # Initialize line for the additional base connection (5 → 9 → 13 → 17)
    base_line_plot_left = ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0]
    base_line_plot_right = ax.plot([], [], [], color="black", linestyle="dashed", linewidth=1)[0]

    # Scatter points for the start of each trajectory (left and right hands)
    start_points_left = {i: ax.scatter([], [], [], color=landmark_colors[i], s=50, zorder=5) for i in landmark_trajectories["Left"]}
    start_points_right = {i: ax.scatter([], [], [], color=landmark_colors[i], s=50, zorder=5) for i in landmark_trajectories["Right"]}

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Hand Landmark 3D Animated Trajectories")

    # Set the camera angle to resemble the video perspective
    ax.view_init(elev=90, azim=90)  # Set elevation and azimuthal angles here
    ax.dist = 8  # Set distance from the viewer to the plot

    max_frames = len(landmark_trajectories["Left"][0]['x'])
    
    def update(frame):
        """Update function for the animation"""
        
        # Check if hands are detected
        left_detected = any(
            landmark_trajectories["Left"][i]["x"][frame] is not None
            for i in landmark_trajectories["Left"]
        ) if landmark_trajectories["Left"] else False
        
        right_detected = any(
            landmark_trajectories["Right"][i]["x"][frame] is not None
            for i in landmark_trajectories["Right"]
        ) if landmark_trajectories["Right"] else False
        
        if not left_detected:
            for line in lines_left.values():
                line.set_data([], [])
                line.set_3d_properties([])
            
            for line in finger_lines_left.values():
                line.set_data([], [])
                line.set_3d_properties([])
                
            for line in wrist_lines_left.values():
                line.set_data([], [])
                line.set_3d_properties([])
            
            base_line_plot_left.set_data([], [])
            base_line_plot_left.set_3d_properties([])
            
            
        if not right_detected:
            for line in lines_right.values():
                line.set_data([], [])
                line.set_3d_properties([])
                
            for line in finger_lines_right.values():
                line.set_data([], [])
                line.set_3d_properties([])
                
            for line in wrist_lines_right.values():
                line.set_data([], [])
                line.set_3d_properties([])
            
            base_line_plot_right.set_data([], [])
            base_line_plot_right.set_3d_properties([])
            

        # Update finger chains for both hands
        for chain in finger_chains:
            if left_detected:
                x_chain = [landmark_trajectories["Left"][i]["x"][frame] for i in chain]
                y_chain = [landmark_trajectories["Left"][i]["y"][frame] for i in chain]
                z_chain = [landmark_trajectories["Left"][i]["z"][frame] for i in chain]
                finger_lines_left[tuple(chain)].set_data(x_chain, y_chain)
                finger_lines_left[tuple(chain)].set_3d_properties(z_chain)
        
            if right_detected:
                x_chain = [landmark_trajectories["Right"][i]["x"][frame] for i in chain]
                y_chain = [landmark_trajectories["Right"][i]["y"][frame] for i in chain]
                z_chain = [landmark_trajectories["Right"][i]["z"][frame] for i in chain]
                finger_lines_right[tuple(chain)].set_data(x_chain, y_chain)
                finger_lines_right[tuple(chain)].set_3d_properties(z_chain)
            

        # Update wrist connections for both hands
        for i in wrist_connections:
            if left_detected:
                x_wrist = [landmark_trajectories["Left"][0]["x"][frame], landmark_trajectories["Left"][i]["x"][frame]]
                y_wrist = [landmark_trajectories["Left"][0]["y"][frame], landmark_trajectories["Left"][i]["y"][frame]]
                z_wrist = [landmark_trajectories["Left"][0]["z"][frame], landmark_trajectories["Left"][i]["z"][frame]]
                wrist_lines_left[i].set_data(x_wrist, y_wrist)
                wrist_lines_left[i].set_3d_properties(z_wrist)
            
            if right_detected:
                x_wrist = [landmark_trajectories["Right"][0]["x"][frame], landmark_trajectories["Right"][i]["x"][frame]]
                y_wrist = [landmark_trajectories["Right"][0]["y"][frame], landmark_trajectories["Right"][i]["y"][frame]]
                z_wrist = [landmark_trajectories["Right"][0]["z"][frame], landmark_trajectories["Right"][i]["z"][frame]]
                wrist_lines_right[i].set_data(x_wrist, y_wrist)
                wrist_lines_right[i].set_3d_properties(z_wrist)


        # Update base connection line (5 → 9 → 13 → 17 for both hands)
        if left_detected:
            x_base_left = [landmark_trajectories["Left"][i]["x"][frame] for i in base_line]
            y_base_left = [landmark_trajectories["Left"][i]["y"][frame] for i in base_line]
            z_base_left = [landmark_trajectories["Left"][i]["z"][frame] for i in base_line]
            base_line_plot_left.set_data(x_base_left, y_base_left)
            base_line_plot_left.set_3d_properties(z_base_left)

        if right_detected:
            x_base_right = [landmark_trajectories["Right"][i]["x"][frame] for i in base_line]
            y_base_right = [landmark_trajectories["Right"][i]["y"][frame] for i in base_line]
            z_base_right = [landmark_trajectories["Right"][i]["z"][frame] for i in base_line]
            base_line_plot_right.set_data(x_base_right, y_base_right)
            base_line_plot_right.set_3d_properties(z_base_right)

        return list(lines_left.values()) + list(lines_right.values()) + list(finger_lines_left.values()) + list(finger_lines_right.values()) + list(wrist_lines_left.values()) + list(wrist_lines_right.values()) + list(start_points_left.values()) + list(start_points_right.values()) + [base_line_plot_left] + [base_line_plot_right]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=True)

    plt.show()

    

if __name__ == "__main__":
    main()
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

# Main func
def main():
    data = load_json('abort')
    trajectories = process_landmarks(data)
    # display_trajectory(trajectories)
    # display_trajectory3d(trajectories)
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
    

# Displays static 2d trajectory from data json via load_json()
def display_trajectory(landmark_trajectories):
    plt.figure(figsize=(8, 6))
    for i, trajectory in landmark_trajectories.items():
        # Plot the trajectory
        plt.plot(trajectory["x"], trajectory["y"], 
                 color=landmark_colors[i], label=f"Landmark {i}" if i in [0, 1, 5, 9, 13, 17] else "", linewidth=1)
        
        # Add the starting point
        plt.scatter(trajectory["x"][0], trajectory["y"][0], 
                    color=landmark_colors[i], s=50, zorder=5, label=f"Start {i}" if i in [0, 1, 5, 9, 13, 17] else "")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Hand Landmark 2D Trajectories")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.axis("equal")  # Ensure aspect ratio is correct
    plt.show()

# Displays static 3d trajectory from data json via load_json()
def display_trajectory3d(landmark_trajectories):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, trajectory in landmark_trajectories.items():
        # Plot the 3D trajectory line
        ax.plot(trajectory["x"], trajectory["y"], trajectory["z"], 
                color=landmark_colors[i], label=f"Landmark {i}" if i in [0, 1, 5, 9, 13, 17] else "", linewidth=1)
        
        # Add the starting point
        ax.scatter(trajectory["x"][0], trajectory["y"][0], trajectory["z"][0], 
                   color=landmark_colors[i], s=50, zorder=5, label=f"Start {i}" if i in [0, 1, 5, 9, 13, 17] else "")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Hand Landmark 3D Trajectories")
    ax.legend(loc="upper right", fontsize="small", ncol=2)

    plt.show()

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

    # Expand limits slightly for better visibility
    margin = 0.1  # 10% margin
    x_range = (x_max - x_min) * margin
    y_range = (y_max - y_min) * margin
    z_range = (z_max - z_min) * margin

    ax.set_xlim(x_min - x_range, x_max + x_range)
    ax.set_ylim(y_min - y_range, y_max + y_range)
    ax.set_zlim(z_min - z_range, z_max + z_range)

    # Initialize empty line objects for each landmark
    lines = {i: ax.plot([], [], [], color=landmark_colors[i], linewidth=1)[0] for i in landmark_trajectories}
    
    # Scatter points for the start of each trajectory
    start_points = {i: ax.scatter([], [], [], color=landmark_colors[i], s=50, zorder=5) for i in landmark_trajectories}

    # Set axis labels and formatting
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Hand Landmark 3D Animated Trajectories")

    # Get total frames (max time steps available)
    max_frames = max(len(traj["frame"]) for traj in landmark_trajectories.values())

    def update(frame):
        """Update function for the animation"""
        for i, trajectory in landmark_trajectories.items():
            # Select the slice of trajectory up to the current frame
            x_data = trajectory["x"][:frame]
            y_data = trajectory["y"][:frame]
            z_data = trajectory["z"][:frame]

            # Update line path
            lines[i].set_data(x_data, y_data)
            lines[i].set_3d_properties(z_data)

            # Keep the start point fixed
            if frame == 1:
                start_points[i]._offsets3d = (np.array([x_data[0]]), np.array([y_data[0]]), np.array([z_data[0]]))

        return list(lines.values()) + list(start_points.values())

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=True)

    plt.show()

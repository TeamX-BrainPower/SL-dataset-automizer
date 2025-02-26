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
    data = load_json('liveFeed')
    landmark_trajectories = process_landmarks(data)
    display_dynamic(landmark_trajectories, data)
    

# Load and return file_name.json
def load_json(file_name):
    with open(f'data/1-raw/{file_name}.json') as f:
        data = json.load(f)
    return data

# Pose landmark connections for visualization
pose_connections = [
    # Face
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    # Body
    [9, 10],  # Mouth
    [11, 12],  # Shoulders
    [23, 24],  # Hips
    [11, 23], [12, 24],  # Hips to legs
    # Arms
    # [9, 11], [10, 12],  # Torso sides
    [11, 13], [13, 15], [15, 17], [17, 19], [19, 21],  # Left arm and hand
    [12, 14], [14, 16], [16, 18], [18, 20], [20, 22],  # Right arm and hand
    # Legs
    # [23, 25], [25, 27], [27, 29], [29, 31],  # Left leg
    # [24, 26], [26, 28], [28, 30], [30, 32]   # Right leg
]

# Colors for pose visualization
pose_color = "cyan"

def calculate_face_center(pose_landmarks):
    # Use nose point (index 0) as face center
    if pose_landmarks and len(pose_landmarks) > 0:
        return {
            "x": pose_landmarks[0]["x"],
            "y": pose_landmarks[0]["y"],
            "z": pose_landmarks[0]["z"]
        }
    return None

def process_landmarks(data):
    landmark_trajectories = {
        "Left": {},
        "Right": {}
    }
    length = data["total_frame_count"]
    
    # Scale factor to match hand depth with pose depth
    # You may need to adjust this value based on your data
    hand_depth_scale = 5.0
    
    for frame in data["frameData"]:
        face_center = None
        if "pose" in frame and frame["pose"]:
            pose_landmarks = frame["pose"][0]["landmarks"]
            face_center = calculate_face_center(pose_landmarks)
            
            # Process pose landmarks relative to face center
            if face_center:
                for i, landmark in enumerate(pose_landmarks):
                    landmark["x"] -= face_center["x"]
                    landmark["y"] -= face_center["y"]
                    landmark["z"] -= face_center["z"]
        
        # Get pose wrist positions
        left_wrist_pos = None
        right_wrist_pos = None
        
        if "pose" in frame and frame["pose"]:
            pose_landmarks = frame["pose"][0]["landmarks"]
            left_wrist_pos = pose_landmarks[15] if len(pose_landmarks) > 15 else None
            right_wrist_pos = pose_landmarks[16] if len(pose_landmarks) > 16 else None
        
        if "hands" not in frame:
            continue
            
        for hand in frame["hands"]:
            hand_marks = hand["landmarks"]
            hand_key = hand["handedness"]
            
            wrist_pos = left_wrist_pos if hand_key == "Left" else right_wrist_pos
            
            if wrist_pos and face_center:
                # Scale the z-coordinates of hand landmarks
                for mark in hand_marks:
                    mark["z"] *= hand_depth_scale
                
                # Calculate offset between hand wrist and pose wrist
                offset_x = wrist_pos["x"] - (hand_marks[0]["x"] - face_center["x"])
                offset_y = wrist_pos["y"] - (hand_marks[0]["y"] - face_center["y"])
                offset_z = wrist_pos["z"] - (hand_marks[0]["z"] - face_center["z"])
                
                for i, landmark in enumerate(hand_marks):
                    if i not in landmark_trajectories[hand_key]:
                        landmark_trajectories[hand_key][i] = {"x": [None] * length, "y": [None] * length, "z": [None] * length}
                    
                    landmark_trajectories[hand_key][i]["x"][frame["frame"]] = landmark["x"] - face_center["x"] + offset_x
                    landmark_trajectories[hand_key][i]["y"][frame["frame"]] = landmark["y"] - face_center["y"] + offset_y
                    landmark_trajectories[hand_key][i]["z"][frame["frame"]] = landmark["z"] - face_center["z"] + offset_z
            else:
                # If no pose wrist position or face center, store centered coordinates
                for i, landmark in enumerate(hand_marks):
                    if i not in landmark_trajectories[hand_key]:
                        landmark_trajectories[hand_key][i] = {"x": [None] * length, "y": [None] * length, "z": [None] * length}
                    
                    # Center coordinates relative to face center if available
                    x_offset = face_center["x"] if face_center else 0
                    y_offset = face_center["y"] if face_center else 0
                    z_offset = face_center["z"] if face_center else 0
                    
                    landmark_trajectories[hand_key][i]["x"][frame["frame"]] = landmark["x"] - x_offset
                    landmark_trajectories[hand_key][i]["y"][frame["frame"]] = landmark["y"] - y_offset
                    landmark_trajectories[hand_key][i]["z"][frame["frame"]] = landmark["z"] - z_offset
                
    return landmark_trajectories


# Displays dynamic 3d trajectory from data json via load_json()
def display_dynamic(landmark_trajectories, data, interval=50):
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

        # Initialize pose lines
    pose_lines = {tuple(conn): ax.plot([], [], [], color=pose_color, linewidth=2)[0] 
                 for conn in pose_connections}
    # pose_points = ax.scatter([], [], [], color=pose_color, s=50, zorder=5)

    # Modify the update function to include pose visualization
    def update(frame):
        # Keep existing hand visualization code
        
        # Add pose visualization
        frame_data = next((f for f in data["frameData"] if f["frame"] == frame), None)
        if frame_data and "pose" in frame_data and frame_data["pose"]:
            pose_landmarks = frame_data["pose"][0]["landmarks"]
            
            # Update pose connections
            for conn in pose_connections:
                start, end = conn
                if start < len(pose_landmarks) and end < len(pose_landmarks):
                    x_pose = [pose_landmarks[start]["x"], pose_landmarks[end]["x"]]
                    y_pose = [pose_landmarks[start]["y"], pose_landmarks[end]["y"]]
                    z_pose = [pose_landmarks[start]["z"], pose_landmarks[end]["z"]]
                    pose_lines[tuple(conn)].set_data(x_pose, y_pose)
                    pose_lines[tuple(conn)].set_3d_properties(z_pose)
        else:
            # Clear pose visualization if no data
            for line in pose_lines.values():
                line.set_data([], [])
                line.set_3d_properties([])

        return (list(lines_left.values()) + list(lines_right.values()) + 
                list(finger_lines_left.values()) + list(finger_lines_right.values()) + 
                list(wrist_lines_left.values()) + list(wrist_lines_right.values()) + 
                list(start_points_left.values()) + list(start_points_right.values()) + 
                [base_line_plot_left, base_line_plot_right] + 
                list(pose_lines.values()))
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Hand Landmark 3D Animated Trajectories")

    # Set the camera angle to resemble the video perspective
    ax.view_init(elev=90, azim=90)  # Set elevation and azimuthal angles here
    ax.dist = 8  # Set distance from the viewer to the plot

    # Adjust axis limits to be symmetric around (0,0,0)
    max_range = max(abs(x_max), abs(x_min), abs(y_max), abs(y_min), abs(z_max), abs(z_min))
    margin = max_range * 0.1  # 10% margin
    ax.set_xlim(-max_range - margin, max_range + margin)
    ax.set_ylim(-max_range - margin, max_range + margin)
    ax.set_zlim(-max_range - margin, max_range + margin)

    max_frames = data["total_frame_count"]
    
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

        # Add pose visualization
        frame_data = next((f for f in data["frameData"] if f["frame"] == frame), None)
        if frame_data and "pose" in frame_data and frame_data["pose"]:
            pose_landmarks = frame_data["pose"][0]["landmarks"]
            
            # Update pose connections
            for conn in pose_connections:
                start, end = conn
                if start < len(pose_landmarks) and end < len(pose_landmarks):
                    x_pose = [pose_landmarks[start]["x"], pose_landmarks[end]["x"]]
                    y_pose = [pose_landmarks[start]["y"], pose_landmarks[end]["y"]]
                    z_pose = [pose_landmarks[start]["z"], pose_landmarks[end]["z"]]
                    pose_lines[tuple(conn)].set_data(x_pose, y_pose)
                    pose_lines[tuple(conn)].set_3d_properties(z_pose)
            
            # Update pose points
            x_points = [lm["x"] for lm in pose_landmarks]
            y_points = [lm["y"] for lm in pose_landmarks]
            z_points = [lm["z"] for lm in pose_landmarks]
        else:
            # Clear pose visualization if no data
            for line in pose_lines.values():
                line.set_data([], [])
                line.set_3d_properties([])

        return (list(lines_left.values()) + list(lines_right.values()) + 
                list(finger_lines_left.values()) + list(finger_lines_right.values()) + 
                list(wrist_lines_left.values()) + list(wrist_lines_right.values()) + 
                list(start_points_left.values()) + list(start_points_right.values()) + 
                [base_line_plot_left, base_line_plot_right] + 
                list(pose_lines.values()))
    
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=True)

    plt.show()

if __name__ == "__main__":
    main()
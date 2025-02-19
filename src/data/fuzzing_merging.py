import json
import numpy as np
import copy
import os


def _add_noise_to_landmarks(landmarks, noise_std=0.01):
    """Add Gaussian noise to each landmark coordinate."""
    noisy_landmarks = []
    for lm in landmarks:
        noisy_lm = {
            "x": lm["x"] + np.random.normal(0, noise_std),
            "y": lm["y"] + np.random.normal(0, noise_std),
            "z": lm["z"] + np.random.normal(0, noise_std)
        }
        noisy_landmarks.append(noisy_lm)
    return noisy_landmarks


def _mirror_landmarks(landmarks):
    """Mirror the x-coordinate of landmarks. Adjust the mirroring factor if needed."""
    mirrored = []
    for lm in landmarks:
        # Assuming x is normalized between 0 and 1, mirror by subtracting from 1.
        mirrored.append({
            "x": 1 - lm["x"],
            "y": lm["y"],
            "z": lm["z"]
        })
    return mirrored


def _augment_frame_data(frame_data, noise_std=0.01, do_mirror=False):
    """Augment frame data by fuzzing each hand's landmarks."""
    augmented_frames = []
    for frame in frame_data:
        new_frame = copy.deepcopy(frame)
        if "hands" in new_frame and new_frame["hands"]:
            for hand in new_frame["hands"]:
                # Add noise to landmarks
                hand["landmarks"] = _add_noise_to_landmarks(hand["landmarks"], noise_std)
                # Optionally mirror landmarks
                if do_mirror:
                    hand["landmarks"] = _mirror_landmarks(hand["landmarks"])
        augmented_frames.append(new_frame)
    return augmented_frames


def _augment_sample(sample, noise_std=0.01, do_mirror=False):
    """Augment a single JSON sample representing a word's data."""
    augmented_sample = copy.deepcopy(sample)
    augmented_sample["frameData"] = _augment_frame_data(sample["frameData"], noise_std, do_mirror)
    return augmented_sample


def generate_augmented_samples(input_filepath, output_dir, num_augmentations=5, noise_std=0.01, do_mirror=False):
    """Generate multiple augmented copies of a JSON sample and save them."""
    # Load the original JSON sample
    with open(input_filepath, "r") as infile:
        data = json.load(infile)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    augmented_samples = []
    for i in range(num_augmentations):
        augmented_data = _augment_sample(data, noise_std, do_mirror)
        augmented_samples.append(augmented_data)
        # Optionally save each augmentation separately
        out_path = os.path.join(output_dir, f"{data['word']}_augmented_{i}.json")
        with open(out_path, "w") as outfile:
            json.dump(augmented_data, outfile, indent=4)

    merged_output = {
        "word": data["word"],
        "instances": augmented_samples
    }
    merged_path = os.path.join(output_dir, f"{data['word']}_merged_augmented.json")
    with open(merged_path, "w") as mfile:
        json.dump(merged_output, mfile, indent=4)

    print(f"Generated {num_augmentations} augmented samples and merged file saved to {output_dir}")

# Sign Language Dataset Automizer
This project automates the processing of sign language videos to extract facial and hand landmarks using MediaPipe. It is designed to create datasets for sign language recognition tasks by processing videos, extracting landmarks, and saving the results in JSON or TFRecord formats.

## Features
- Video Processing: Processes sign language videos to extract frames.
- Landmark Detection: Detects facial and hand landmarks using MediaPipe's FaceMesh and HandLandmarker.
- Data Export: Saves extracted landmarks in JSON or TFRecord formats.
- Visualization: Displays annotated frames with landmarks and FPS for debugging and visualization.

## Requirements
- Python 3.12 or higher
- MediaPipe
- OpenCV
- TensorFlow (optional, for TFRecord support)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SL-dataset-automizer.git
   cd SL-dataset-automizer
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. **Configure the Project:**

   Modify the src/config.py file to set your preferences for displaying output, saving JSON, or saving TFRecords.

2. **Run the Script:**
    
    Execute the main.py script to process a video:
    ```bash
    python src/main.py
    ```
    By default, the script processes a video for the word "abort" from a predefined URL. You can modify the video_url and word variables in main.py to process other videos.

3. **Output:**
- Processed landmarks are saved in the output directory. 
- If display_output is enabled, the annotated video is displayed in a window.

## Project Structure
```
SL-dataset-automizer/
├── src/
│   ├── config.py                # Configuration settings
│   ├── main.py                  # Entry point for the script
│   ├── video_processor.py       # Handles video processing and landmark detection
│   ├── visualization/
│   │   └── drawer.py            # Utilities for drawing landmarks on frames
│   ├── data/
│   │   └── processors.py        # Handles data processing and saving (JSON, TFRecord)
│   └── models/
│       ├── face_landmarker.task # Pre-built face landmarker model by Mediapipe  
│       ├── hand_landmarker.task # Pre-built hand landmarker model by Mediapipe
│       └── landmarker.py        # Factory for creating MediaPipe landmarkers
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

# License
This project is licensed under the MIT License. See the LICENSE file for details.


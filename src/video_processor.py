import cv2
import time
import mediapipe as mp
from pathlib import Path

from config import ProcessingConfig, MotionDetectionConfig 
from data.processors import JSONProcessor, TFRecordProcessor
from models.landmarker import LandmarkerFactory
from visualization.drawer import LandmarkDrawer
from collections import deque


class MotionDetection:
    def __init__(self, config: MotionDetectionConfig):
        self.config = config 

        self.left_wrist_points = []
        self.right_wrist_points = []
        self.detection_window_size = config.detection_window_size # Detection window size in frames
        self.threshold_delta = config.threshold_delta # Threshold for stationary wrists
        self.stationary_count = 0
        
        # Variables to track velocity
        self.left_velocity = None
        self.right_velocity = None
        
        # Counters for tracking missing hands
        self.left_hand_missing_count = 0
        self.right_hand_missing_count = 0
        self.left_hand_detected = False
        self.right_hand_detected = False
        self.missing_threshold = config.missing_threshold  # Number of frames before considering hand truly missing
        
        # Flag to ensure we have stable detection before analyzing pauses
        self.stable_detection_count = 0
        self.min_stable_frames = config.min_stable_frames  # Minimum number of frames needed before checking for pauses
        self.has_displayed_pause = False  # To avoid repeated pause messages

        self.start_frame = 0
        self.end_frame = 0
        self.is_in_motion = False

    def detect_motion(self, hand_result, frame_count):
        if hand_result and hand_result.hand_landmarks:
            for landmarks, handedness in zip(
                    hand_result.hand_landmarks,
                    hand_result.handedness
            ):
                if handedness[0].category_name == "Left":
                    self.left_hand_detected = True
                    self.left_hand_missing_count = 0  # Reset counter when hand is detected
                    
                    left_wrist = landmarks[0]
                    average_left_wrist = (left_wrist.x + left_wrist.y + left_wrist.z) / 3
                    self.left_wrist_points.append(average_left_wrist)
                    
                    # Calculate sliding window velocity once we have enough points
                    if len(self.left_wrist_points) >= self.detection_window_size:
                        # Calculate average velocity over the window
                        window = self.left_wrist_points[-self.detection_window_size:]
                        velocities = [window[i+1] - window[i] for i in range(len(window)-1)]
                        self.left_velocity = sum(velocities) / len(velocities) if velocities else 0
                        
                        # Trim the list to prevent it from growing too large
                        if len(self.left_wrist_points) > 30:
                            self.left_wrist_points = self.left_wrist_points[-30:]
                            
                elif handedness[0].category_name == "Right":
                    self.right_hand_detected = True
                    self.right_hand_missing_count = 0  # Reset counter when hand is detected
                    
                    right_wrist = landmarks[0]
                    average_right_wrist = (right_wrist.x + right_wrist.y + right_wrist.z) / 3
                    self.right_wrist_points.append(average_right_wrist)
                    
                    # Calculate sliding window velocity for right wrist
                    if len(self.right_wrist_points) >= self.detection_window_size:
                        window = self.right_wrist_points[-self.detection_window_size:]
                        velocities = [window[i+1] - window[i] for i in range(len(window)-1)]
                        self.right_velocity = sum(velocities) / len(velocities) if velocities else 0
                        
                        if len(self.right_wrist_points) > 30:
                            self.right_wrist_points = self.right_wrist_points[-30:]

                else:
                    raise ValueError("Handedness not detected")
            
            # Handle missing hands
            if not self.left_hand_detected:
                self.left_hand_missing_count += 1
                if self.left_hand_missing_count >= self.missing_threshold:
                    self.left_velocity = 0.0
            
            if not self.right_hand_detected:
                self.right_hand_missing_count += 1
                if self.right_hand_missing_count >= self.missing_threshold:
                    self.right_velocity = 0.0
            
            # Increment stable detection counter if we detected at least one hand
            if self.left_hand_detected or self.right_hand_detected:
                self.stable_detection_count += 1
            
            # Only check for pauses if we have sufficient stable detection
            if self.stable_detection_count >= self.min_stable_frames:
                # Only perform pause detection if we have velocity data for both hands
                # or if one hand has been missing long enough to have zero velocity
                if (self.left_velocity is not None or self.left_hand_missing_count >= self.missing_threshold) and \
                    (self.right_velocity is not None or self.right_hand_missing_count >= self.missing_threshold):
                    
                    # For missing hands, assume velocity is zero
                    actual_left_vel = self.left_velocity if self.left_velocity is not None else 0.0
                    actual_right_vel = self.right_velocity if self.right_velocity is not None else 0.0
                    
                    if abs(actual_left_vel) <= self.threshold_delta and abs(actual_right_vel) <= self.threshold_delta:
                        self.stationary_count += 1
                        if self.stationary_count >= 3 and not self.has_displayed_pause:
                            print("Signer has paused")
                            print("frame: ",frame_count)
                            self.has_displayed_pause = True
                            if self.is_in_motion:
                                self.end_frame = frame_count
                                self.is_in_motion = False
                            return [self.start_frame, self.end_frame]
                    else:
                        # vi va ikke i bevegelse før -> nå e vi i bevegelse
                        if not self.is_in_motion:
                            self.start_frame = frame_count
                            self.is_in_motion = True
                        # Reset counters when movement is detected
                        self.stationary_count = 0
                        self.has_displayed_pause = False
                        return None
                else:
                    # Not enough data yet
                    self.stationary_count = 0
        else:
            # No hands detected at all
            self.left_hand_missing_count += 1
            self.right_hand_missing_count += 1
            
            # Reset stable detection if no hands are visible for too long
            if self.left_hand_missing_count >= self.missing_threshold and self.right_hand_missing_count >= self.missing_threshold:
                self.stable_detection_count = 0
            
            if self.left_hand_missing_count >= self.missing_threshold:
                self.left_velocity = 0.0
            
            if self.right_hand_missing_count >= self.missing_threshold:
                self.right_velocity = 0.0
            
            # Reset stationary counter when no hands are detected
            self.stationary_count = 0
            self.has_displayed_pause = False

        return None







class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.face_landmarker = None
        self.hand_landmarker = None
        self.temp_count = 0

        motion_config = MotionDetectionConfig()

        self.motion_detector = MotionDetection(motion_config)

        self.processors = []

    def process_video(self, video_path = 0, word = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")


        # Process frames
        with LandmarkerFactory.create_landmarkers(self.config) as (
                self.face_landmarker,
                self.hand_landmarker):
            self._process_frames(cap, [], word)

        
        # Save outputs 
        #self._save_outputs(cap, word)


      

    def _process_frames(self, cap, processors, word):
        frame_count = 0
        prev_time = time.time()
        queue = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            mp_image = self._prepare_frame(frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Detect landmarks
            face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            queue.append([face_result, hand_result])

            # Detect motion 
            motion_data = self.motion_detector.detect_motion(hand_result, frame_count)
            if motion_data:
                print(f"Motion detected between frames {motion_data[0]} and {motion_data[1]}")
                
                

            

            # Update processors
            for processor in processors:
                processor.process_frame(hand_result, frame_count)


            # Display output if configured
            if self.config.display_output:
                self._display_frame(mp_image, face_result, hand_result, prev_time)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

      
            frame_count += 1
            prev_time = time.time()

        cap.release()
        if self.config.display_output:
            cv2.destroyAllWindows()

    def _prepare_frame(self, frame):
        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _display_frame(self, mp_image, face_result, hand_result, prev_time):
        annotated_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        annotated_image = LandmarkDrawer.draw_landmarks(
            annotated_image,
            face_result,
            hand_result
        )

        # Add FPS counter
        fps = 1 / (time.time() - prev_time)
        cv2.putText(
            annotated_image,
            f'FPS: {int(fps)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow('Landmarker', annotated_image)

    def _save_outputs(self, cap, word):
        Path(self.config.output_dir).mkdir(exist_ok=True)

        # Initialize processors
        processors = []
        if self.config.save_json:
            json_processor = JSONProcessor(
                word,
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FPS))
            )
            processors.append(json_processor)

        if self.config.save_tfrecord:
            tfrecord_processor = TFRecordProcessor(word)
            processors.append(tfrecord_processor)


        if self.config.save_json:
            if word is None:
                self.temp_count += 1
                json_processor.save(f"{self.config.output_dir}/temp{self.temp_count}.json")
            else: 
                json_processor.save(
                    f"{self.config.output_dir}/{word}.json"
                )
        if self.config.save_tfrecord:
            for processor in processors:
                processor.save(f"{self.config.output_dir}/{processor.word}.tfrecord")
        

    


    
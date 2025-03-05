import cv2
import time
import mediapipe as mp
from pathlib import Path

from config import ProcessingConfig
from data.processors import JSONProcessor, TFRecordProcessor
from models.landmarker import LandmarkerFactory
from visualization.drawer import LandmarkDrawer


class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.face_landmarker = None
        self.hand_landmarker = None

    def process_video(self, video_path = 0, word = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

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

        # Process frames
        with LandmarkerFactory.create_landmarkers(self.config) as (
                self.face_landmarker,
                self.hand_landmarker
        ):
            self._process_frames(cap, processors)


        # Save outputs
        Path(self.config.output_dir).mkdir(exist_ok=True)
        if self.config.save_json:
            json_processor.save(
                f"{self.config.output_dir}/{word}.json"
            )
        if self.config.save_tfrecord:
            tfrecord_processor.save(
                f"{self.config.output_dir}/{word}.tfrecord"
            )

    def _process_frames(self, cap, processors):
        frame_count = 0
        prev_time = time.time()

        left_wrist_points = []
        right_wrist_points = []
        dt = 5 # Detection window time in frames
        threshold_velocity = 0.000995 # Threshold for stationary wrists
        stationary_count = 0
        
        # Variables to track velocity
        left_velocity = None
        right_velocity = None
        
        # Counters for tracking missing hands
        left_hand_missing_count = 0
        right_hand_missing_count = 0
        missing_threshold = 10  # Number of frames before considering hand truly missing
        
        # Flag to ensure we have stable detection before analyzing pauses
        stable_detection_count = 0
        min_stable_frames = 10  # Minimum number of frames needed before checking for pauses
        has_displayed_pause = False  # To avoid repeated pause messages

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

            # Update processors
            for processor in processors:
                processor.process_frame(hand_result, frame_count)

            # Flag to track if hands were detected in this frame
            left_hand_detected = False
            right_hand_detected = False

            #####################  Detection window #####################
            if hand_result and hand_result.hand_landmarks:
                for landmarks, handedness in zip(
                        hand_result.hand_landmarks,
                        hand_result.handedness
                ):
                    if handedness[0].category_name == "Left":
                        left_hand_detected = True
                        left_hand_missing_count = 0  # Reset counter when hand is detected
                        
                        left_wrist = landmarks[0]
                        average_left_wrist = (left_wrist.x + left_wrist.y + left_wrist.z) / 3
                        left_wrist_points.append(average_left_wrist)
                        
                        # Calculate sliding window velocity once we have enough points
                        if len(left_wrist_points) >= dt:
                            # Calculate average velocity over the window
                            window = left_wrist_points[-dt:]
                            velocities = [window[i+1] - window[i] for i in range(len(window)-1)]
                            left_velocity = sum(velocities) / len(velocities) if velocities else 0
                            
                            # Trim the list to prevent it from growing too large
                            if len(left_wrist_points) > 30:
                                left_wrist_points = left_wrist_points[-30:]
                                
                    elif handedness[0].category_name == "Right":
                        right_hand_detected = True
                        right_hand_missing_count = 0  # Reset counter when hand is detected
                        
                        right_wrist = landmarks[0]
                        average_right_wrist = (right_wrist.x + right_wrist.y + right_wrist.z) / 3
                        right_wrist_points.append(average_right_wrist)
                        
                        # Calculate sliding window velocity for right wrist
                        if len(right_wrist_points) >= dt:
                            window = right_wrist_points[-dt:]
                            velocities = [window[i+1] - window[i] for i in range(len(window)-1)]
                            right_velocity = sum(velocities) / len(velocities) if velocities else 0
                            
                            if len(right_wrist_points) > 30:
                                right_wrist_points = right_wrist_points[-30:]

                    else:
                        raise ValueError("Handedness not detected")
                
                # Handle missing hands
                if not left_hand_detected:
                    left_hand_missing_count += 1
                    if left_hand_missing_count >= missing_threshold:
                        left_velocity = 0.0
                
                if not right_hand_detected:
                    right_hand_missing_count += 1
                    if right_hand_missing_count >= missing_threshold:
                        right_velocity = 0.0
                
                # Increment stable detection counter if we detected at least one hand
                if left_hand_detected or right_hand_detected:
                    stable_detection_count += 1
                
                # Only check for pauses if we have sufficient stable detection
                if stable_detection_count >= min_stable_frames:
                    # Only perform pause detection if we have velocity data for both hands
                    # or if one hand has been missing long enough to have zero velocity
                    if (left_velocity is not None or left_hand_missing_count >= missing_threshold) and \
                       (right_velocity is not None or right_hand_missing_count >= missing_threshold):
                        
                        # For missing hands, assume velocity is zero
                        actual_left_vel = left_velocity if left_velocity is not None else 0.0
                        actual_right_vel = right_velocity if right_velocity is not None else 0.0
                        
                        if abs(actual_left_vel) <= threshold_velocity and abs(actual_right_vel) <= threshold_velocity:
                            stationary_count += 1
                            if stationary_count >= 3 and not has_displayed_pause:
                                print("Signer has paused")
                                has_displayed_pause = True
                        else:
                            # Reset counters when movement is detected
                            stationary_count = 0
                            has_displayed_pause = False
                    else:
                        # Not enough data yet
                        stationary_count = 0
            else:
                # No hands detected at all
                left_hand_missing_count += 1
                right_hand_missing_count += 1
                
                # Reset stable detection if no hands are visible for too long
                if left_hand_missing_count >= missing_threshold and right_hand_missing_count >= missing_threshold:
                    stable_detection_count = 0
                
                if left_hand_missing_count >= missing_threshold:
                    left_velocity = 0.0
                
                if right_hand_missing_count >= missing_threshold:
                    right_velocity = 0.0
                
                # Reset stationary counter when no hands are detected
                stationary_count = 0
                has_displayed_pause = False

            #####################  Detection Window end  #####################

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

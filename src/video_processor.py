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
        self.gesture_recognizer = None
        self.pose_landmarker = None

    def process_video(self, video_path = 0 , word: str= None):

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
                self.hand_landmarker,
                self.gesture_recognizer,
                self.pose_landmarker
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            mp_image = self._prepare_frame(frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Detect landmarks
            # face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Update processors
            for processor in processors:
                # processor.process_hands(hand_result, frame_count)
                # # processor.process_face(face_result, frame_count)
                # processor.process_gesture(gesture_result, frame_count)
                # processor.process_pose(pose_result, frame_count)
                processor.process_frame(frame_count, hand_result, None, gesture_result, pose_result)

            # Display output if configured
            if self.config.display_output:
                self._display_frame(mp_image, None, hand_result, gesture_result, pose_result, prev_time)
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

    def _display_frame(self, mp_image, face_result, hand_result, gesture_result, pose_result,  prev_time):
        annotated_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        annotated_image = LandmarkDrawer.draw_landmarks(
            annotated_image,
            # face_result,
            None,
            hand_result, 
            gesture_result, 
            pose_result
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
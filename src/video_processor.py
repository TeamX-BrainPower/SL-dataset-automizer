from typing import Optional
import cv2
import time
import mediapipe as mp
from pathlib import Path

from config import ProcessingConfig

# from data.processors import JSONProcessor, TFRecordProcessor
from models.landmarker import LandmarkerFactory
from visualization.drawer import LandmarkDrawer


class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.face_landmarker = None
        self.hand_landmarker = None
        self.gesture_recognizer = None
        self.pose_landmarker = None

    def process_video(
        self,
        video_path: str | int,
        word: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        if isinstance(video_path, str) and "youtube" in video_path:
            import pafy

            if start_frame is None:
                raise ValueError("Start frame is None. Needs to be a valid frame")

            try:
                video = pafy.new(video_path, ydl_opts={"nocheckcertificate": True})
                best = video.getbest()
                video_path = best.url  # pyright: ignore
            except Exception as e:
                print(f"Error processing video: {e}")
                return
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if not cap.isOpened():
                raise ValueError(f"Could not open video stream: {video_path}")
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video stream: {video_path}")

        # Initialize processors
        processors = []
        json_processor = None
        tfrecord_processor = None
        if self.config.save_json:
            from data.processors import JSONProcessor

            json_processor = JSONProcessor(
                word,
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FPS)),
            )
            processors.append(json_processor)

        if self.config.save_tfrecord:
            from data.processors import TFRecordProcessor

            tfrecord_processor = TFRecordProcessor(word)
            processors.append(tfrecord_processor)

        # Process frames
        with LandmarkerFactory.create_landmarkers(self.config) as (
            self.face_landmarker,
            self.hand_landmarker,
            self.gesture_recognizer,
            self.pose_landmarker,
        ):
            self._process_frames(cap, processors, end_frame)

        # Save outputs
        Path(self.config.output_dir).mkdir(exist_ok=True)
        if self.config.save_json and json_processor is not None:
            json_processor.save(f"{self.config.output_dir}/{word}.json")
        if self.config.save_tfrecord and tfrecord_processor is not None:
            tfrecord_processor.save(f"{self.config.output_dir}/{word}.tfrecord")

    def _process_frames(self, cap, processors, end_frame=None):
        frame_count = 0
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Check if we've reached the end frame or the video has ended
            if not ret or (end_frame is not None and current_frame >= end_frame):
                break

            # Process frame
            mp_image = self._prepare_frame(frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Detect landmarks
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)  # pyright: ignore
            gesture_result = self.gesture_recognizer.recognize_for_video(  # pyright: ignore
                mp_image, timestamp_ms
            )
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)  # pyright: ignore

            # Update processors
            for processor in processors:
                # processor.process_hands(hand_result, frame_count)
                # # processor.process_face(face_result, frame_count)
                # processor.process_gesture(gesture_result, frame_count)
                # processor.process_pose(pose_result, frame_count)
                processor.process_frame(
                    frame_count, hand_result, None, gesture_result, pose_result
                )

            # Display output if configured
            if self.config.display_output:
                self._display_frame(
                    mp_image, None, hand_result, gesture_result, pose_result, prev_time
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            prev_time = time.time()

        cap.release()
        if self.config.display_output:
            cv2.destroyAllWindows()

    def _prepare_frame(self, frame):
        # flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _display_frame(
        self, mp_image, face_result, hand_result, gesture_result, pose_result, prev_time
    ):
        annotated_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        annotated_image = LandmarkDrawer.draw_landmarks(
            annotated_image,
            # face_result,
            None,
            hand_result,
            gesture_result,
            pose_result,
        )

        # Add FPS counter
        fps = 1 / (time.time() - prev_time)
        cv2.putText(
            annotated_image,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Landmarker", annotated_image)

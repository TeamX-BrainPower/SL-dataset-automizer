import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import time
from config import ProcessingConfig
from video_processor import VideoProcessor


class SignLanguageRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recorder")

        # Initialize all instance attributes
        self.config = ProcessingConfig()
        self.video_processor = VideoProcessor(self.config)
        self.recording = False  # Initialize recording flag
        self.countdown_active = False  # Initialize countdown flag
        self.frame_buffer = []  # Initialize frame buffer
        self.last_movement_time = 0  # Initialize movement timer
        self.video_running = False  # Initialize video running flag

        self.setup_ui()
        self.setup_video()

    def setup_ui(self):
        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # Right panel controls
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=tk.N)

        # Word input
        ttk.Label(right_panel, text="Enter sign word:").pack(pady=5)
        self.word_entry = ttk.Entry(right_panel, width=20)
        self.word_entry.pack(pady=5)

        # Recording button
        self.record_btn = ttk.Button(right_panel, text="Start Recording",
                                     command=self.start_recording)
        self.record_btn.pack(pady=20)

        # Instructions
        instructions = ("When pressing this button, hold your hands down at your waist."
                        "When the counter reaches 0, perform the sign."
                        "The video will stop when your hands stop moving"
                        "or after 2 seconds has passed.")
        ttk.Label(right_panel, text=instructions, wraplength=200).pack(pady=10)

        # Status label
        self.status_label = ttk.Label(right_panel, text="", foreground="blue")
        self.status_label.pack(pady=10)

    def setup_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.destroy()
            return

        self.video_running = True
        self.update_video()

    def update_video(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                if self.recording:
                    self.frame_buffer.append(frame)
                    self.detect_movement(frame)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_video)

    def start_recording(self):
        if not self.word_entry.get():
            messagebox.showwarning("Input Error", "Please enter a word first")
            return

        self.countdown_active = True
        self.record_btn.config(state=tk.DISABLED)
        self.start_countdown(3)

    def start_countdown(self, count):
        if count >= 0:
            self.status_label.config(text=str(count), foreground="red")
            self.root.after(1000, self.start_countdown, count - 1)
        else:
            self.countdown_active = False
            self.status_label.config(text="Recording...", foreground="green")
            self.start_recording_session()

    def start_recording_session(self):
        self.recording = True
        self.last_movement_time = time.time()
        self.frame_buffer = []

        # Reinitialize landmarker for new session
        self.video_processor.create_new_landmarker()
        self.root.after(self.config.recording_timeout * 1000, self.stop_recording)

    def detect_movement(self, frame):
        if not self.video_processor.hand_landmarker:
            return

        mp_image = self.video_processor._prepare_frame(frame)

        # Correct detection method for Tasks API
        hand_result = self.video_processor.hand_landmarker.detect_for_video(
            mp_image,
            int(time.time() * 1000)  # Timestamp in ms
        )

        if hand_result.hand_landmarks:
            self.last_movement_time = time.time()

        if (time.time() - self.last_movement_time) > self.config.movement_timeout:
            self.stop_recording()

    def stop_recording(self):
        self.recording = False
        self.record_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Recording saved for: " + self.word_entry.get())

        # Process and save the recording using existing VideoProcessor
        self.process_recording()

    def process_recording(self):
        word = self.word_entry.get()
        video_path = self._save_frames_to_video()

        try:
            # Process the video
            self.video_processor.process_video(video_path, word)
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            # Ensure the file is closed before deletion
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"File '{video_path}' deleted successfully.")
                except PermissionError:
                    print(f"Failed to delete '{video_path}': File is still in use.")
                except FileNotFoundError:
                    print(f"File '{video_path}' not found.")

    def _save_frames_to_video(self):
        if not self.frame_buffer:
            print("No frames to save!")
            return None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp_video.mp4", fourcc, 30.0, self.config.frame_size)

        for frame in self.frame_buffer:
            out.write(frame)

        out.release()
        return "temp_video.mp4"

    def on_close(self):
        self.video_running = False
        self.cap.release()
        if self.video_processor.hand_landmarker:
            self.video_processor.hand_landmarker.close()
        self.root.destroy()

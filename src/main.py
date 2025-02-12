from config import ProcessingConfig
from video_processor import VideoProcessor
from gui_app import SignLanguageRecorderApp
import tkinter as tk


def main():
    config = ProcessingConfig(
        new_sign_recorder=True,
        display_output=False,
        save_json=False,
        save_tfrecord=False
    )

    if config.new_sign_recorder:
        root = tk.Tk()
        app = SignLanguageRecorderApp(root)
        root.mainloop()
    else:
        processor = VideoProcessor(config)

        # Process a single video
        for word in ["abort", "kaos", "melke", "sex", "skriver", "skyve", "vin"]:
            video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
            processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

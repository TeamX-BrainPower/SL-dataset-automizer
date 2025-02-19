from config import ProcessingConfig
from src.data.fuzzing_merging import generate_augmented_samples
from video_processor import VideoProcessor
from gui_app import SignLanguageRecorderApp
import tkinter as tk


def main():
    config = ProcessingConfig(
        new_sign_recorder=True,
        display_output=False,
        save_json=True,
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

        if ProcessingConfig.handle_data:
            generate_augmented_samples(f"data/1-raw/{word}.json", "data/2-processed",
                                       num_augmentations=5, noise_std=0.02, do_mirror=True)


if __name__ == "__main__":
    main()

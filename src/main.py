# from time import time
from config import ProcessingConfig
from grid_subscriber import GridSubscriber
from live_processor import LiveProsessor

# from video_processor import VideoProcessor
from mediapipe.tasks.python.vision import RunningMode


def main():
    config = ProcessingConfig(
        display_output=False,
        save_json=True,
        save_tfrecord=True,
        vision_mode=RunningMode.VIDEO,
    )

    processor = LiveProsessor(config)

    grid_subscriber = GridSubscriber()

    processor.add_subscriber(grid_subscriber)

    processor.process()

    # Process a single video
    # for word in ["abort", "kaos", "melke", "sex", "skriver", "skyve", "vin"]:
    #     video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    #     processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

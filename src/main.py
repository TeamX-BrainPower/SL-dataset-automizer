from src.config import ProcessingConfig
from src.video_processor import VideoProcessor


def main():
    config = ProcessingConfig(
        display_output=True,
        save_json=True,
        save_tfrecord=True
    )

    processor = VideoProcessor(config)

    # Process a single video
    word = "abort"
    video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

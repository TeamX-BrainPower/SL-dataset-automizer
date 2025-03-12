from config import ProcessingConfig
from video_processor import VideoProcessor


def main():
    config = ProcessingConfig(
        display_output=True,
        save_json=True,
        save_tfrecord=False
    )

    processor = VideoProcessor(config)
    webprocessor = VideoProcessor(config)

    # Process webcam stream
    webprocessor.process_video(0, "livefeed")

    # Process a video file
    word = "skilsmisse"
    video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    processor.process_video(video_url, word)

if __name__ == "__main__":
    main()

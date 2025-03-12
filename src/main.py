from config import ProcessingConfig
from video_processor import VideoProcessor


def main():
    config = ProcessingConfig(
        display_output=True,
        save_json=True,
        save_tfrecord=False
    )


    # Process webcam stream
    webprocessor = VideoProcessor(config)
    webprocessor.process_video()

    # Process a video file
    #processor = VideoProcessor(config)
    #word = "flaskeaapner"
    #video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    #processor.process_video(video_url, word)
    """ word = ["flaskeaapner", "katt", "barnefilm", "galning"]
    for word in word: 
        print(f"Processing video for word: {word}")
        video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
        processor.process_video(video_url, word)
    """
    

if __name__ == "__main__":
    main()


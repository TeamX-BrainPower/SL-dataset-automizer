from config import ProcessingConfig
from video_processor import VideoProcessor
import json
# from machine_learning import train_model


def main():
    config = ProcessingConfig(display_output=False, save_json=True, save_tfrecord=False)

    processor = VideoProcessor(config)
    # webprocessor = VideoProcessor(config)

    # processor.process_video(
    #     "https://www.youtube.com/watch?v=S2cqitZ0qes", "library", 0, 73
    # )

    with open("data/MS-ASL/MSASL_train.json", "r") as file:
        data = json.load(file)
        for i, entry in enumerate(data):
            try:
                if entry.get("clean_text") != "rainbow":
                    continue
                print(f"Processing Rainbow: {i}")
                url = entry.get("url")
                clean_text = entry.get("clean_text")
                start = entry.get("start")
                end = entry.get("end")

                # Process the video with the extracted information
                processor.process_video(
                    url, clean_text, start_frame=start, end_frame=end
                )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")

    # Train the model
    # print("Training model...")
    # train_model()
    # Process webcam stream
    # webprocessor.process_video(0, "skilsmisse")

    # Process a video file
    # word = "skilsmisse"
    # video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    # processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

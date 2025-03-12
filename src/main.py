from config import ProcessingConfig
from video_processor import VideoProcessor
import json
# from machine_learning import train_model


def main():
    config = ProcessingConfig(display_output=True, save_json=True, save_tfrecord=False)

    processor = VideoProcessor(config)
    webprocessor = VideoProcessor(config)

    # Process videos
    # for word in [
    #                 "hus", "bil-1", "bord", "stol", "vindu-1",
    #                 "bok", "blomst-1", "doer", "lampe", "vei",
    #                 "barn", "hund-1", "katt", "fugl", "fisk",
    #                 "skole", "by-1", "skog-1", "elv", "fjell"
    #             ]:
    #     video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    #     processor.process_video(video_url, word)
    # video_url = "https://www.youtube.com/watch?v=S2cqitZ0qes"
    # processor.process_video(video_url, "library")
    with open("MS-ASL/MSASL_train.json", "r") as file:
        data = json.load(file)
        for entry in data:
            try:
                if entry.get("clean_text") != "library":
                    continue
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
    webprocessor.process_video(0, "livefeed")

    # Process a video file
    word = "skilsmisse"
    video_url = f"https://www.minetegn.no/Tegnordbok-HTML/video_/{word}.mp4"
    processor.process_video(video_url, word)


if __name__ == "__main__":
    main()

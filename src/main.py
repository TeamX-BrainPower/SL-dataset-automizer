from config import ProcessingConfig
from video_processor import VideoProcessor
import json
import os
from absl import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.set_verbosity(logging.ERROR)


def main():
    config = ProcessingConfig(display_output=False, save_json=True, save_tfrecord=False)

    processor = VideoProcessor(config)

    words = ["eat", "nice", "want"]

    with open("data/MS-ASL/MSASL_train.json", "r") as file:
        data = json.load(file)
        for i, entry in enumerate(data):
            try:
                if entry.get("clean_text") not in words:
                    continue
                print(f"Processing {entry.get('clean_text')}: {i}")
                url = entry.get("url")
                clean_text = entry.get("clean_text")
                start = entry.get("start")
                end = entry.get("end")

                # Process the video with the extracted information
                processor.process_video(
                    url, clean_text, start_frame=start, end_frame=end, data_type="train"
                )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")

    with open("data/MS-ASL/MSASL_val.json", "r") as file:
        data = json.load(file)
        for i, entry in enumerate(data):
            try:
                if entry.get("clean_text") not in words:
                    continue
                print(f"Processing {entry.get('clean_text')}: {i}")
                url = entry.get("url")
                clean_text = entry.get("clean_text")
                start = entry.get("start")
                end = entry.get("end")

                # Process the video with the extracted information
                processor.process_video(
                    url, clean_text, start_frame=start, end_frame=end,  data_type="val"
                )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")

    with open("data/MS-ASL/MSASL_test.json", "r") as file:
        data = json.load(file)
        for i, entry in enumerate(data):
            try:
                if entry.get("clean_text") not in words:
                    continue
                print(f"Processing {entry.get('clean_text')}: {i}")
                url = entry.get("url")
                clean_text = entry.get("clean_text")
                start = entry.get("start")
                end = entry.get("end")

                # Process the video with the extracted information
                processor.process_video(
                    url, clean_text, start_frame=start, end_frame=end, data_type="test"
                )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")
            


if __name__ == "__main__":
    main()

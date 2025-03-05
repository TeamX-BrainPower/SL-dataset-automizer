from typing import Any
from config import ProcessingConfig
from pipeline import PipelineComponent
import numpy as np


class RecorderPipeline(PipelineComponent):
    config: ProcessingConfig
    all_data: np.ndarray
    recording_index: int
    headers: list[str]

    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config
        self.all_data = np.array([])
        self.recording_index = 0
        self.headers = ["timestamp"]

        extra = []

        for pose in [
            "nose",
            "right_shoulder",
            "left_shoulder",
            "right_elbow",
            "left_elbow",
            "right_wrist",
            "left_wrist",
        ]:
            for point in ["x", "y", "z"]:
                extra.append(f"{pose}_{point}")

        for hand in ["right", "left"]:
            for i in range(21):
                for point in ["x", "y", "z"]:
                    extra.append(f"{hand}_finger_{i}_{point}")

        self.headers += extra

        self.headers += [f"vel_{point}" for point in extra]
        self.headers += [f"acc_{point}" for point in extra]

        self.headers += [f"distance_{i}" for i in range(27)]

    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return data

        if self.config.recording:
            if len(self.all_data) >= 150:
                self.all_data = np.vstack([self.all_data, data[-1, :]])
            else:
                self.all_data = data
        else:
            if len(self.all_data) > 0:
                np.savetxt(
                    f"{self.config.output_dir}/recording_{self.recording_index}.txt",
                    self.all_data,
                    delimiter=",",
                    header=",".join(self.headers),
                )
                self.recording_index += 1
                self.all_data = np.array([])
        return data

    def get_data(self) -> Any:
        return None

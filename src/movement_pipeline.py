from typing import Any
from pipeline import PipelineComponent
import numpy as np


class MovementPipeline(PipelineComponent):
    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return

        # 0 -> timestamp
        timestamp = data[:, 0]
        positions = data[:, 1:148]
        dt = np.diff(timestamp, axis=0)[:, np.newaxis]

        velocity = np.diff(positions, axis=0) / dt

        if data.shape[0] != velocity.shape[0]:
            velocity = np.vstack([np.zeros((1, velocity.shape[1])), velocity])

        acceleration = np.diff(velocity, axis=0) / dt
        while acceleration.shape[0] != data.shape[0]:
            acceleration = np.vstack(
                [np.zeros((1, acceleration.shape[1])), acceleration]
            )

        return np.hstack([data, velocity, acceleration])

    def get_data(self) -> Any:
        return

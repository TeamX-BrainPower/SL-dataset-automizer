from typing import Any
from scipy.interpolate import interp1d
from pipeline import PipelineComponent
import numpy as np


class InterpolationPipeline(PipelineComponent):
    interpolation_size: int

    def __init__(self, interpolation_size: int = 150) -> None:
        self.interpolation_size = interpolation_size
        return

    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return None

        data_len = len(data)

        if data_len < 150:
            return data

        original_indices = np.linspace(0, 1, data_len)

        new_indicies = np.linspace(0, 1, self.interpolation_size)

        interpolated_frames = np.zeros((self.interpolation_size, data.shape[1]))

        interpolated_frames[:, 0] = data[:, 0]

        for point in range(1, 49 * 3 + 1):
            coord_values = data[:, point]
            interp_func = interp1d(
                original_indices,
                coord_values,
                kind="cubic",
                axis=0,
                fill_value="extrapolate",  # pyright: ignore
            )
            interpolated_frames[:, point] = interp_func(new_indicies)

        return interpolated_frames

    def get_data(self) -> Any:
        return None

from collections import deque
from typing import Deque
from typing_extensions import override

from numpy.typing import NDArray
from subscriber import Subscriber
import numpy as np
from scipy.interpolate import interp1d


class InterpolationSubscriber(Subscriber):
    frames: Deque[NDArray]
    full_array: NDArray
    max_size: int
    interpolation_size: int

    def __init__(self, max_size: int = 60, interpolation_size: int = 30) -> None:
        self.frames = deque(maxlen=max_size)
        self.full_array = np.zeros((max_size, 49, 3))
        self.max_size = max_size
        self.interpolation_size = interpolation_size
        return

    @override
    def consume(self, results: NDArray, *args):
        self.frames.append(results)

        # interpolate if they are larger than 15 and less than 40
        frame_len = len(self.frames)
        # how do we detect motion??

        if 60 > frame_len > 15:
            frames = np.array(self.frames)

            original_indices = np.linspace(0, 1, frame_len)

            new_indices = np.linspace(0, 1, self.interpolation_size)

            interpolated_frames = np.zeros((self.interpolation_size, 49, 3))

            for point in range(49):
                for coord in range(3):
                    coord_values = frames[:, point, coord]
                    interp_func = interp1d(
                        original_indices,
                        coord_values,
                        kind="cubic",
                        axis=0,
                        fill_value="extrapolate",  # pyright: ignore
                    )
                    interpolated_frames[:, point, coord] = interp_func(new_indices)

            print("We have interpolated:", len(interpolated_frames))

        elif frame_len == self.max_size:
            print("we have max size frames")
            pass

        return

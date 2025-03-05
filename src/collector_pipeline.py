from collections import deque
from typing import Any, Deque, Optional
from pipeline import PipelineComponent
import numpy as np


class CollectorPipeline(PipelineComponent):
    buffer: Deque[np.ndarray]

    def __init__(self, max_size: Optional[int] = 150) -> None:
        self.buffer = deque(np.array([], dtype=np.float64), maxlen=max_size)
        return

    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return None

        self.buffer.append(data)

        if len(self.buffer) < 5:
            return None

        return np.array(self.buffer, dtype=np.float64)

    def get_data(self) -> Any:
        if len(self.buffer) < 5:
            return None
        return np.array(self.buffer, dtype=np.float64)

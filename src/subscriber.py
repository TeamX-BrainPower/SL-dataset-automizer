from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Subscriber(ABC):
    @abstractmethod
    def consume(self, results: NDArray, *args): ...

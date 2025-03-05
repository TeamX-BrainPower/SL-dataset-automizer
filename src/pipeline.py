from abc import ABC, abstractmethod
from typing import Any


class PipelineComponent(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any: ...

    @abstractmethod
    def get_data(self) -> Any: ...


class PipelineManager:
    pipeline: list[PipelineComponent]

    def __init__(self) -> None:
        self.pipeline = []

    def add_component(self, component: PipelineComponent) -> None:
        self.pipeline.append(component)

    def process(self, input_data: Any) -> Any:
        data = input_data
        for component in self.pipeline:
            if data is None:
                print("data was None")
                return None
            data = component.process(data)
            if data is None:
                data = component.get_data()

        return data

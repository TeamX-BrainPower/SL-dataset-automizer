# from collections import OrderedDict
from collections import OrderedDict
from numpy.typing import NDArray
from typing_extensions import override
from pipeline import PipelineComponent
import numpy as np


class MaxSizedOrderedDict(OrderedDict):
    max_size: int

    def __init__(self, max_size: int, *args, **kwargs) -> None:
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value) -> None:
        if len(self) >= self.max_size and key not in self:
            self.popitem(last=False)
        super().__setitem__(key, value)


class DataEnsurerPipeline(PipelineComponent):
    incoming_data: OrderedDict[int, dict]
    outgoing_data: OrderedDict[int, dict]

    def __init__(self) -> None:
        self.incoming_data = MaxSizedOrderedDict(150)
        self.outgoing_data = MaxSizedOrderedDict(150)
        return

    # Data is a dict with keys: timestamp, pose or hands, and results
    @override
    def process(self, data: dict) -> None:
        # get if it is pose or hand
        # then for each time we add data check if we have completed a new one

        timestamp = data.pop("timestamp")
        if timestamp not in self.incoming_data:
            self.incoming_data[timestamp] = data
        else:
            if self.incoming_data[timestamp].get("hands"):
                self.incoming_data[timestamp]["results"][:7] = data["results"][:7]
                self.incoming_data[timestamp]["pose"] = True

            elif self.incoming_data[timestamp].get("pose"):
                self.incoming_data[timestamp]["results"][7:] = data["results"][7:]
                self.incoming_data[timestamp]["hands"] = True

            self.incoming_data[timestamp]["results"][7:28] += self.incoming_data[
                timestamp
            ]["results"][5]
            self.incoming_data[timestamp]["results"][28:49] += self.incoming_data[
                timestamp
            ]["results"][6]

            self.outgoing_data[timestamp] = self.incoming_data[timestamp].copy()
        return

    @override
    def get_data(self) -> NDArray | None:
        return_arr = np.array([v["results"] for v in self.outgoing_data.values()])

        if len(return_arr) == 0:
            return None

        return return_arr

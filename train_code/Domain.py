from typing import List
from Source import Source
import torch
import time


class Domain:
    def __init__(self, index) -> None:
        self.index: int = index
        self.type: str = ""
        self.step: int = -1
        self.sourceList: List[Source] = None
        self.data_p: torch.Tensor = None
        self.data_v: torch.Tensor = None
        self.base_propagation: torch.Tensor = None

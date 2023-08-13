from typing import List
from Source import Source
import torch
import time


class Domain:
    def __init__(self, index) -> None:
        self.index: int = index
        self.step: int = -1
        self.source: Source = None
        self.data_p: torch.Tensor = None
        self.data_v: torch.Tensor = None
        self.base_propagation: torch.Tensor = None
        self.propagation_p: torch.Tensor = None
        self.propagation_v: torch.Tensor = None


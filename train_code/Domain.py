from typing import List
from Source import Source
import torch

class Domain:
    def __init__(self, index) -> None:
        self.index: int = index
        self.step: int = -1
        self.source: Source = None
        self.data_p: torch.Tensor = None
        self.data_v: torch.Tensor = None
        self.data_propagation: torch.Tensor = None

    def sourceUpdate(self):
        self.step += 1
        sourceUpdate = self.source.getSourceUpdate()
        self.data_p += sourceUpdate
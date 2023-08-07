import torch
from typing import List


class Domain:
    def __init__(self, index) -> None:
        self.index: int = index
        self.step: int = -1
        self.source: List[torch.Tensor] = None
        self.data_p: torch.Tensor = None
        self.data_v: torch.Tensor = None
        self.data_propagation: torch.Tensor = None

    def hasNoneCheck(self) -> bool:
        attributes = vars(self)
        return any(value is None or value == -1 for value in attributes.values())

    def updateSource(self, source):
        self.source = source

    def updateP(self, data_p):
        self.data_p = data_p

    def updateV(self, data_v):
        self.data_v = data_v

    def updatePropagation(self, data_propagation):
        self.data_propagation = data_propagation


if __name__ == "__main__":
    domain = Domain(0)
    print(domain.hasNoneCheck())
    print(domain)

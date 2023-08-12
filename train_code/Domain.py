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
        self.data_propagation: torch.Tensor = None

    def update(self):
        self.step += 1
        sourceExpression = self.source.sourceExpression
        change = sourceExpression[self.step % len(sourceExpression)]
        sourceMask = self.source.sourceMask

        self.data_p[
            :,
            sourceMask.leftTopX : sourceMask.leftTopX + sourceMask.sourceWidth,
            sourceMask.leftTopY : sourceMask.leftTopY + sourceMask.sourceHeight,
        ] = change

        self.data_v[
            0:1,
            sourceMask.leftTopX + 1 : sourceMask.leftTopX + sourceMask.sourceWidth,
            sourceMask.leftTopY : sourceMask.leftTopY + sourceMask.sourceHeight,
        ] = 0

        self.data_v[
            1:2,
            sourceMask.leftTopX : sourceMask.leftTopX + sourceMask.sourceWidth,
            sourceMask.leftTopY + 1 : sourceMask.leftTopY + sourceMask.sourceHeight,
        ] = 0

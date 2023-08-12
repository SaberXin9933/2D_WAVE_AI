import numpy as np
from typing import List


class SourceMask:
    def __init__(self, leftTopX, leftTopY, sourceWidth, sourceHeight, mask) -> None:
        self.leftTopX = leftTopX
        self.leftTopY = leftTopY
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.mask = mask


class Source:
    def __init__(self, sourceMask: SourceMask, sourceExpression: np.array) -> None:
        self.sourceMask: SourceMask = sourceMask
        self.sourceExpression: np.array = sourceExpression

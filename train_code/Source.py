import numpy as np


class Source:
    def __init__(self, x,y, width, height, mask, sourceExpression: np.array) -> None:
        self.x = x
        self.y = y
        self.sourceWidth = width
        self.sourceHeight = height
        self.mask = mask
        self.sourceExpression: np.array = sourceExpression

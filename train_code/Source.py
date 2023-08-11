import numpy as np
from typing import List


class SourcePoint:
    def __init__(self, x, y, val) -> None:
        self.x = x
        self.y = y
        self.val = val


class Source:
    def __init__(
        self, sourcePointList: List[SourcePoint], sourceExpression: np.array
    ) -> None:
        self.stepRecord = 0
        self.sourcePointList: List[SourcePoint] = sourcePointList
        self.sourceExpression: np.array = sourceExpression

    def getSourceUpdate(self) -> List[SourcePoint]:
        nowStep = self.stepRecord
        change = self.sourceExpression[nowStep % len(self.sourceExpression)]
        for i in range(len(self.sourcePointList)):
            self.sourcePointList[i].val = change
        self.stepRecord += 1
        return self.sourcePointList

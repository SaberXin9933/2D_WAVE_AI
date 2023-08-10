import numpy as np


class Source:
    def __init__(self, sourceMask: np.array, sourceExpression: np.array) -> None:
        self.stepRecord = 0
        self.sourceMask: np.array = sourceMask
        self.sourceExpression: np.array = sourceExpression

    def getSourceUpdate(self):
        if self.stepRecord > 100:
            return 0
        nextStep = self.stepRecord + 1
        nowStep = self.stepRecord
        change = (
            self.sourceExpression[nextStep % len(self.sourceExpression)]
            - self.sourceExpression[nowStep % len(self.sourceExpression)]
        )

        self.stepRecord += 1
        return change * self.sourceMask

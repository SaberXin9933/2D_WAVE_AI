import numpy as np
from matplotlib import pyplot as plt


class Source:
    def __init__(self, sourceMask: np.array, sourceExpression: np.array) -> None:
        self.stepRecord = 0
        self.sourceMask: np.array = sourceMask
        self.sourceExpression: np.array = sourceExpression

    def getSourceUpdate(self):
        nextStep = (self.stepRecord + 1) % len(self.sourceExpression)
        nowStep = (self.stepRecord) % len(self.sourceExpression)
        change = self.sourceExpression[nextStep] - \
            self.sourceExpression[nowStep]

        self.stepRecord += 1
        return change * self.sourceMask

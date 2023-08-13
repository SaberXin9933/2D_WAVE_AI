import numpy as np
import torch
from Source import Source
from Context import Context
from typing import List
from utils.mathUtils import sigmodNP


class SourceManager:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = params = context.params
        self.domainWidth = params.domainWidth
        self.domainHeight = params.domainHeight
        self.whRate = params.whRate
        self.boundaryRate = params.boundaryRate
        self.pointNumber = params.pointNumber
        self.decay_point = params.decay_point
        self.minT = params.minT
        self.maxT = params.maxT
        self.minBiasRate = params.minBiasRate
        self.maxBiasRate = params.maxBiasRate
        self.dtype = params.dtype

    def getRandomSouce(self) -> Source:
        x, y, sourceWH, mask = self.getRandomMask()
        sourceExpression = self.getRandomSourceExpression(
            self.pointNumber,
            self.decay_point,
            self.minT,
            self.maxT,
            self.minBiasRate,
            self.maxBiasRate,
        )

        return Source(x, y, sourceWH, sourceWH, mask, sourceExpression)

    """
    掩码
    """

    def getRandomMask(self) -> tuple:
        sourceWHRate = self.whRate * np.random.rand()
        sourceWH = 2 + int(sourceWHRate * min(self.domainWidth, self.domainHeight))

        leftTopXRate = self.boundaryRate + np.random.rand() * (
            1 - self.boundaryRate * 2 - self.whRate
        )
        leftTopX = int(self.domainWidth * leftTopXRate)

        leftTopYRate = self.boundaryRate + np.random.rand() * (
            1 - self.boundaryRate * 2 - self.whRate
        )
        leftTopY = int(self.domainHeight * leftTopYRate)

        mask = torch.ones(1, sourceWH, sourceWH)

        return (leftTopX, leftTopY, sourceWH, mask)

    """
    表达式
    """

    def getSourceExpression(
        self, pointNumber: int, decay_point: int, T: int, biasRate: float
    ) -> np.array:
        bias = T * biasRate
        swx = np.arange(pointNumber) + bias
        swy = np.sin(swx * np.pi * 2 / T)
        data = swy

        decay_line = np.arange(decay_point) * (12 / decay_point)
        decay_line = decay_line * (12 / np.max(decay_line))
        decay_line = decay_line - np.max(decay_line) / 2
        decay_line = sigmodNP(decay_line)
        decay_line = np.concatenate(
            (
                decay_line,
                np.ones(pointNumber - 2 * decay_point),
                np.flip(decay_line),
            )
        )
        return decay_line * data

    def getRandomSourceExpression(
        self,
        pointNumber: int,
        decay_point: int,
        minT: int,
        maxT: int,
        minBiasRate: int,
        maxBiasRate: int,
    ) -> np.array:
        T = np.random.randint(minT, maxT)
        biasRate = minBiasRate + np.random.rand() * (maxBiasRate - minBiasRate)
        return self.getSourceExpression(
            pointNumber=pointNumber, decay_point=decay_point, T=T, biasRate=biasRate
        )


# 测试随机声源
def test1():
    from matplotlib import pyplot as plt
    from Context import Params

    params = Params()
    params.domainWidth = 200
    params.domainHeight = 200
    params.whRate = 0.07
    params.boundaryRate = 0.2
    params.pointNumber = 100
    params.minT = 20
    params.maxT = 100
    params.minBiasRate = 0.3
    params.maxBiasRate = 0.7
    context = Context(params)
    sourceManager = SourceManager(context)
    source: Source = sourceManager.getRandomSouce()

    # plt.matshow(source.sourceMask)
    # plt.colorbar()

    swx = np.arange(source.sourceExpression.size)
    plt.scatter(swx, source.sourceExpression)

    plt.tight_layout()  # This helps prevent overlapping of subplots
    plt.show()


def test2():
    import torch, time

    x = torch.arange(40000).reshape((200, 200))
    t1 = time.time()
    y = torch.arange(400).reshape((20, 20))

    x[100:120, 100:120] = y
    print(time.time() - t1)


if __name__ == "__main__":
    test2()

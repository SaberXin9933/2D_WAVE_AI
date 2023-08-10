import numpy as np
import torch
from Source import Source
from Context import Context


class SourceManager:
    def __init__(self, context: Context) -> None:
        self.params = params = context.params
        self.domainWidth = params.domainWidth
        self.domainHeight = params.domainHeight
        self.whRate = params.whRate
        self.boundaryRate = params.boundaryRate
        self.pointNumber = params.pointNumber
        self.minT = params.minT
        self.maxT = params.maxT
        self.minBiasRate = params.minBiasRate
        self.maxBiasRate = params.maxBiasRate
        self.dtype = params.dtype

    def getRandomSouce(self):
        sourceMask = torch.Tensor(
            self.getRandomSourceMask(
                self.domainWidth, self.domainHeight, self.whRate, self.boundaryRate
            )
        ).to(self.params.dtype)
        sourceExpression = torch.Tensor(
            self.getRandomSourceExpression(
                self.pointNumber,
                self.minT,
                self.maxT,
                self.minBiasRate,
                self.maxBiasRate,
            )
        ).to(self.params.dtype)
        return Source(sourceMask, sourceExpression)

    def getSourceMask(
        self,
        domainWidth: int,
        domainHeight: int,
        centerX: int,
        centerY: int,
        sourceWidth: int,
        sourceHeight: int,
    ) -> np.array:
        shape = (domainWidth, domainHeight)
        mask = np.zeros(shape)
        mask[
            centerX - sourceWidth // 2 : centerX + sourceWidth // 2,
            centerY - sourceHeight // 2 : centerY + sourceHeight // 2,
        ] = 1
        return mask

    def getRandomSourceMask(
        self, domainWidth: int, domainHeight: int, whMaxRate: float, boundaryRate: float
    ):
        sourceWHRate = whMaxRate * np.random.rand()
        sourceWH = 2 + int(sourceWHRate * min(domainWidth, domainHeight))

        centerXRate = boundaryRate + np.random.rand() * (1 - boundaryRate * 2)
        centerX = int(domainWidth * centerXRate)

        centerYRate = boundaryRate + np.random.rand() * (1 - boundaryRate * 2)
        centerY = int(domainHeight * centerYRate)

        return self.getSourceMask(
            domainWidth, domainHeight, centerX, centerY, sourceWH, sourceWH
        )

    def getSourceExpression(self, pointNumber: int, T: int, biasRate: int) -> np.array:
        bias = int(pointNumber * biasRate)
        swx = np.arange(pointNumber)
        swy = np.sin(swx * np.pi * 2 / T)
        rate = np.exp(-((swx - bias) ** 2) / (pointNumber**2 / 15))
        data = swy * rate
        max_data = np.max(np.abs(data))
        return data / max_data

    def getRandomSourceExpression(
        self, pointNumber: int, minT: int, maxT: int, minBiasRate: int, maxBiasRate: int
    ) -> np.array:
        T = np.random.randint(minT, maxT)
        biasRate = minBiasRate + np.random.rand() * (maxBiasRate - minBiasRate)
        return self.getSourceExpression(pointNumber=pointNumber, T=T, biasRate=biasRate)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    w = 200
    h = 200
    whRate = 0.07
    boundaryRate = 0.2
    pointNumber = 200
    minT = 30
    maxT = 200
    minBiasRate = 0.3
    maxBiasRate = 0.7
    sourceManager = SourceManager(
        w, h, whRate, boundaryRate, pointNumber, minT, maxT, minBiasRate, maxBiasRate
    )
    source: Source = sourceManager.getRandomSouce()

    for i in range(100):
        print(source.stepRecord, np.max(np.abs(source.getSourceUpdate())))

    # plt.matshow(source.sourceMask)
    # plt.colorbar()

    swx = np.arange(pointNumber)
    plt.scatter(swx, source.sourceExpression)

    plt.tight_layout()  # This helps prevent overlapping of subplots
    plt.show()

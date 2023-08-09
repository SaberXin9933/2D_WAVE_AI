import numpy as np
from Source import Source


class SourceManager:
    def __init__(self,w:int ,h:int,whRate:float,boundaryRate:float,pointNumber:int,minT:int,maxT:int,minBiasRate:float,maxBiasRate:float) -> None:
        self.domainWidth = w
        self.domainHeight = h
        self.whRate = whRate
        self.boundaryRate = boundaryRate
        self.pointNumber = pointNumber
        self.minT = minT
        self.maxT = maxT
        self.minBiasRate = minBiasRate
        self.maxBiasRate = maxBiasRate

    def getRandomSouce(self):
        sourceMask = self._getRandomSourceMask(self.domainWidth, self.domainHeight, self.whRate, self.boundaryRate)
        sourceExpression = self._getRandomSourceExpression(self.pointNumber, self.minT, self.maxT, self.minBiasRate, self.maxBiasRate)
        return Source(sourceMask, sourceExpression)

    def _getSourceMask(self, domainWidth: int, domainHeight: int, centerX: int, centerY: int, sourceWidth: int, sourceHeight: int) -> np.array:
        shape = (domainWidth,domainHeight)
        mask = np.zeros(shape)
        mask[centerX-sourceWidth//2:centerX+sourceWidth//2,
             centerY-sourceHeight//2:centerY+sourceHeight//2] = 1
        return mask

    def _getRandomSourceMask(self, domainWidth: int, domainHeight: int, whMaxRate: float, boundaryRate: float):
        sourceWHRate = whMaxRate*np.random.rand()
        sourceWH = 2+int(sourceWHRate*min(domainWidth,domainHeight))

        centerXRate = boundaryRate + np.random.rand()*(1-boundaryRate*2)
        centerX = int(domainWidth*centerXRate)

        centerYRate = boundaryRate+np.random.rand()*(1-boundaryRate*2)
        centerY = int(domainHeight*centerYRate)

        return self._getSourceMask(domainWidth, domainHeight, centerX, centerY, sourceWH, sourceWH)

    def _getSourceExpression(self, pointNumber: int, T: int, biasRate: int) -> np.array:
        bias = int(pointNumber*biasRate)
        swx = np.arange(pointNumber)
        swy = np.sin(swx*np.pi*2/T)
        rate = np.exp(-(swx-bias)**2/(pointNumber**2/15))
        data = swy*rate
        max_data = np.max(np.abs(data))
        return data/max_data

    def _getRandomSourceExpression(self, pointNumber: int, minT: int, maxT: int, minBiasRate: int, maxBiasRate: int) -> np.array:
        T = np.random.randint(minT, maxT)
        biasRate = minBiasRate + np.random.rand()*(maxBiasRate-minBiasRate)
        return self._getSourceExpression(pointNumber=pointNumber, T=T, biasRate=biasRate)


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
    sourceManager = SourceManager(w,h,whRate,boundaryRate,pointNumber,minT,maxT,minBiasRate,maxBiasRate)
    source:Source = sourceManager.getRandomSouce()

    for i in range(100):
        print(source.stepRecord,np.max(np.abs(source.getSourceUpdate())))


    # plt.matshow(source.sourceMask)
    # plt.colorbar()

    swx = np.arange(pointNumber)
    plt.scatter(swx, source.sourceExpression)

    plt.tight_layout()  # This helps prevent overlapping of subplots
    plt.show()


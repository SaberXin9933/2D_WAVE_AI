import numpy as np
from matplotlib import pyplot as plt
from Source import Source
from Params import Params


class SourceManager:
    def randomInitByParams(self, params: Params):
        sourceMask = self.getRandomSourceMask(
            params.domainWidth, params.domainHeight, params.WHRate, params.boundaryRate)
        sourceExpression = self.getRandomSourceExpression(
            params.pointNumber, params.minT, params.maxT, params.minBiasRate, params.maxBiasRate)
        return Source(sourceMask, sourceExpression)

    def getSourceMask(self, domainWidth: int, domainHeight: int, centerX: int, centerY: int, sourceWidth: int, sourceHeight: int) -> np.array:
        mask = np.zeros(domainWidth, domainHeight)
        mask[centerX-sourceWidth//2:centerX+sourceWidth//2,
             centerY-sourceHeight//2:centerY+sourceHeight//2] = 1
        return mask

    def getRandomSourceMask(self, domainWidth: int, domainHeight: int, WHRate: float, boundaryRate: float):
        widthRate = WHRate + np.random.rand()*(1-WHRate*2)
        sourceWidth = int(domainWidth*widthRate)

        heightRate = WHRate + np.random.rand()*(1-WHRate*2)
        sourceHeight = int(domainWidth*heightRate)

        centerXRate = boundaryRate + np.random.rand()*(1-boundaryRate*2)
        centerX = int(domainWidth*centerXRate)

        centerYRate = boundaryRate+np.random.rand()*(1-boundaryRate*2)
        centerY = int(domainHeight*centerYRate)

        return self.getSourceMask(domainWidth, domainHeight, centerX, centerY, sourceWidth, sourceHeight)

    def getSourceExpression(self, pointNumber: int, T: int, biasRate: int) -> np.array:
        bias = int(pointNumber*biasRate)
        swx = np.arange(pointNumber)
        swy = np.sin(swx*np.pi*2/T)
        rate = np.exp(-(swx-bias)**2/(pointNumber**2/15))
        data = swy*rate
        max_data = np.max(np.abs(data))
        return data/max_data

    def getRandomSourceExpression(self, pointNumber: int, minT: int, maxT: int, minBiasRate: int, maxBiasRate: int) -> np.array:
        T = np.random.randint(minT, maxT)
        biasRate = minBiasRate + np.random.rand()*(maxBiasRate-minBiasRate)
        return self.getSourceExpression(pointNumber=pointNumber, T=T, biasRate=biasRate)

    def testSingle(self):
        pointNumber = 200
        T = 50
        biasRate = 0.5
        swx = np.arange(pointNumber)
        sourceExpression = self.getSourceExpression(pointNumber, T, biasRate)
        plt.scatter(swx, sourceExpression)
        plt.show()

    def testRandom(self):
        pointNumber = 200
        minT = 30
        maxT = 200
        minBiasRate = 0.3
        maxBiasRate = 0.7

        swx = np.arange(pointNumber)
        sourceExpression = self.getRandomSourceExpression(
            pointNumber, minT, maxT, minBiasRate, maxBiasRate)
        plt.scatter(swx, sourceExpression)
        plt.show()


fSourceManager = SourceManager()
if __name__ == "__main__":
    fSourceManager.testRandom()

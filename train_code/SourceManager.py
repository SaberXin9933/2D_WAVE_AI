import numpy as np
import torch
from Source import Source
from Context import Context
from typing import List
from utils.mathUtils import sigmodNP
import random


class SourceManager:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.log = context.logger
        params = context.params
        self.domainWidth = params.domainWidth
        self.domainHeight = params.domainHeight
        self.boundaryWH = params.boundaryWH
        self.propagationWidth = params.domainWidth - 2 * params.boundaryWH
        self.propagationHeight = params.domainHeight - 2 * params.boundaryWH
        self.pointNumber = params.pointNumber
        self.decay_point = params.decay_point
        self.minT = params.minT
        self.maxT = params.maxT
        self.minBiasRate = params.minBiasRate
        self.maxBiasRate = params.maxBiasRate
        self.minSouceWH = params.minSouceWH
        self.maxSouceWH = params.maxSouceWH
        self.cellWH = params.cellWH
        self.maxSourceNum = params.maxSourceNum
        self.dtype = params.dtype

        self.initPartition()

    """传播域分区"""

    def initPartition(self):
        # 计算传播域分区信息
        pw = self.propagationWidth
        ph = self.propagationHeight
        cellWH = self.cellWH
        boundaryWH = self.boundaryWH

        w_partition_times = pw // cellWH
        h_partition_times = ph // cellWH

        partition_result = []
        for i in range(w_partition_times):
            for j in range(h_partition_times):
                x = i * cellWH + boundaryWH
                y = j * cellWH + boundaryWH
                w = pw - i * cellWH if i == w_partition_times - 1 else cellWH
                h = ph - j * cellWH if j == h_partition_times - 1 else cellWH
                partition_result.append((x, y, w, h))
        self.maxSourceNum = min(len(partition_result), self.maxSourceNum)
        self.partition_result = partition_result

    """获取源项"""

    def getSourceBySet(self, T, biasRate, sourceX, sourcepY, sourceWH):
        sourceExpression = self.getSourceExpression(
            pointNumber=self.pointNumber,
            decay_point=self.decay_point,
            T=T,
            biasRate=biasRate,
        )
        mask = torch.ones(1, sourceWH, sourceWH)
        return Source(sourceX, sourcepY, sourceWH, sourceWH, mask, sourceExpression)

    def getRandomSourceList(self, sourceNum: int = None) -> List[Source]:
        sourceNum = random.randint(1, self.maxSourceNum)
        if self.maxSourceNum < sourceNum:
            self.log.error(
                f"声源点数量超出范围,已经调整为最大声源数量,PRE:{sourceNum},NOW:{self.maxSourceNum}"
            )
            sourceNum = self.maxSourceNum

        partitionList = random.sample(self.partition_result, sourceNum)
        sourceList = []
        for partition in partitionList:
            sourceExpression = self.getRandomSourceExpression()
            cellX, cellY, cellW, cellH = partition
            sourceX, sourcepY, sourceWH = self.getSourceDistribution(
                cellX, cellY, cellW, cellH
            )
            mask = torch.ones(1, sourceWH, sourceWH)
            source = Source(
                sourceX, sourcepY, sourceWH, sourceWH, mask, sourceExpression
            )
            sourceList.append(source)
        return sourceList

    """分配声源位置"""

    def getSourceDistribution(self, cellX, cellY, cellW, cellH) -> tuple:
        maxSouceWH = self.maxSouceWH
        minSouceWH = self.minSouceWH
        sourceWH = random.randint(minSouceWH, maxSouceWH)

        leftTopX = cellX + random.randint(0, cellW - sourceWH)
        leftTopY = cellY + random.randint(0, cellH - sourceWH)

        return (leftTopX, leftTopY, sourceWH)

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
    ) -> np.array:
        T = np.random.randint(self.minT, self.maxT)
        biasRate = self.minBiasRate + np.random.rand() * (
            self.maxBiasRate - self.minBiasRate
        )
        return self.getSourceExpression(
            pointNumber=self.pointNumber,
            decay_point=self.decay_point,
            T=T,
            biasRate=biasRate,
        )


"""测试"""

# 测试随机声源
def test1():
    from matplotlib import pyplot as plt
    from Context import Params

    params = Params()
    context = Context(params)
    sourceManger = SourceManager(context)
    sourceList = sourceManger.getRandomSourceList(150)
    source = sourceList[0]
    print(sourceList)

    # plt.matshow(source.sourceMask)
    # plt.colorbar()

    swx = np.arange(source.sourceExpression.size)
    plt.scatter(swx, source.sourceExpression)

    plt.tight_layout()  # This helps prevent overlapping of subplots
    plt.show()

    plt.matshow(source.mask.squeeze().cpu().numpy())
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    test1()

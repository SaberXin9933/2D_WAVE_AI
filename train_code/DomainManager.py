from Domain import Domain
import torch
import numpy as np
from SourceManager import SourceManager

class DomainManager:
    def __init__(self,w:int ,h:int, sourceManager:SourceManager) -> None:
        self.w = w
        self.h = h
        self.sourceManager = sourceManager

    '''获取随机正方形传播域'''
    def getRandomField(self):
        w,h = self.w,self.h
        t = np.random.rand()*100000000
        x_mesh, y_mesh = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
        x_mesh, y_mesh = 1.0 * x_mesh, 1.0 * y_mesh
        data = np.sin(x_mesh * 0.02 + np.cos(y_mesh * 0.01 * (np.cos(t * 0.0021) + 2)) * np.cos(t * 0.01) * 3 + np.cos(
            x_mesh * 0.011 * (np.sin(t * 0.00221) + 2)) * np.cos(t * 0.00321) * 3 + 0.01 * y_mesh * np.cos(t * 0.0215))
        data = (5 + data/torch.max(data))/5
        data /= torch.mean(data)
        return data

    '''获取随机计算域'''
    def getRandomDomain(self,index:int)->Domain:
        domain = Domain(index)
        domain.data_p = torch.zeros(1,self.w,self.h)
        domain.data_v = torch.zeros(2,self.w,self.h)
        domain.data_propagation = self.getRandomField()
        domain.source = self.sourceManager.getRandomSouce()
        return domain

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    index:int = 0
    w:int=200
    h:int = 200
    whRate:float = 0.07
    boundaryRate:float = 0.2
    pointNumber = 200
    minT = 30
    maxT = 200
    minBiasRate = 0.3
    maxBiasRate = 0.7

    sourceManager = SourceManager(w,h,whRate, boundaryRate,pointNumber, minT, maxT, minBiasRate, maxBiasRate)
    mananger = DomainManager(w,h,sourceManager)
    domain = mananger.getRandomDomain(index)
    plt.matshow(domain.data_propagation)
    plt.colorbar()
    plt.show()
from Domain import Domain
import torch
import numpy as np
from SourceManager import SourceManager
from Context import Context, Params


class DomainManager:
    def __init__(self, context: Context, sourceManager: SourceManager = None) -> None:
        self.params = context.params
        self.w = self.params.domainWidth
        self.h = self.params.domainHeight
        self.dytpe = self.params.dtype
        self.sourceManager = (
            sourceManager if sourceManager != None else SourceManager(context)
        )

    """获取随机正方形传播域"""

    def getRandomField(self):
        w, h = self.w, self.h
        t = np.random.rand() * 100000000
        x_mesh, y_mesh = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
        x_mesh, y_mesh = 1.0 * x_mesh, 1.0 * y_mesh
        data = np.sin(
            x_mesh * 0.02
            + np.cos(y_mesh * 0.01 * (np.cos(t * 0.0021) + 2)) * np.cos(t * 0.01) * 3
            + np.cos(x_mesh * 0.011 * (np.sin(t * 0.00221) + 2))
            * np.cos(t * 0.00321)
            * 3
            + 0.01 * y_mesh * np.cos(t * 0.0215)
        )
        data = (1000 + data / torch.max(data)) / 1000
        data /= torch.max(data)
        # return data.to(self.dytpe).unsqueeze(0) * self.getPMLField()
        return self.getPMLField()

    """获取随机计算域"""

    def getRandomDomain(self, index: int) -> Domain:
        domain = Domain(index)
        domain.data_p = torch.zeros(1, self.w, self.h).to(self.dytpe)
        domain.data_v = torch.zeros(2, self.w, self.h).to(self.dytpe)
        domain.data_propagation = self.getRandomField().to(self.dytpe)
        domain.source = self.sourceManager.getRandomSouce()
        return domain

    """获取PML层"""

    def getPMLField(self):
        field_line = torch.flip(
            1
            - np.log(1 / (0.000001))
            * (1.5 * self.params.c / self.params.pml_width)
            * (torch.arange(self.params.pml_width) / self.params.pml_width),
            dims=[0],
        )
        field_line /= torch.max(field_line)
        xField = torch.ones(1, self.w, self.h).to(self.dytpe)
        yField = torch.ones(1, self.w, self.h).to(self.dytpe)
        xField[:, : self.params.pml_width, :] *= (
            (field_line).unsqueeze(0).unsqueeze(2)
        )
        xField[:, -self.params.pml_width :, :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(2)
        )
        yField[:, :, : self.params.pml_width] *= (
            (field_line).unsqueeze(0).unsqueeze(1)
        )
        yField[:, :, -self.params.pml_width :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(1)
        )
        return torch.min(xField,yField)


def test1():
    from matplotlib import pyplot as plt

    index: int = 0
    w: int = 200
    h: int = 200
    whRate: float = 0.07
    boundaryRate: float = 0.2
    pointNumber = 200
    minT = 30
    maxT = 200
    minBiasRate = 0.3
    maxBiasRate = 0.7
    params = Params()
    context = Context()
    sourceManager = SourceManager(context)
    mananger = DomainManager(context)
    domain = mananger.getRandomDomain(index)
    plt.matshow(domain.data_propagation)
    plt.colorbar()
    plt.show()


def test2():
    from matplotlib import pyplot as plt

    context = Context()
    mananger = DomainManager(context)
    pmlField = mananger.getPMLField()
    plt.matshow(pmlField.squeeze().numpy())
    plt.colorbar()
    plt.show()

def test3():
    from matplotlib import pyplot as plt

    context = Context()
    mananger = DomainManager(context)
    pmlField = mananger.getPMLField()
    line = pmlField[:,100,:].squeeze().numpy()
    plt.scatter(range(len(line)),line)
    plt.show()


if __name__ == "__main__":
    test2()

from Domain import Domain
import torch
import numpy as np
from SourceManager import SourceManager
from FinitDiffenceManager import FinitDiffenceManager
from Derivatives import Derivatives
from Context import Context, Params


class DomainManager:
    def __init__(
        self,
        context: Context,
    ) -> None:
        self.params = context.params
        self.w = self.params.domainWidth
        self.h = self.params.domainHeight
        self.dytpe = self.params.dtype
        self.sourceManager = SourceManager(context)
        self.derivatives = Derivatives(
            self.params.kernel_point_number,
            self.params.kernel_order,
            self.params.kernel_delta,
            kernel_dtype=self.params.dtype,
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
        return data.to(self.dytpe).unsqueeze(0)

    """获取计算域"""

    def getDomain(self, index: int) -> Domain:
        params = self.params
        if params.type == "test" and params.testIsRandom == True:
            return self.getSpecifiedDomain(index)
        else:
            return self.getRandomDomain(index)

    def getRandomDomain(self, index: int) -> Domain:
        domain = Domain(index)
        domain.data_p = torch.zeros(1, self.w, self.h).to(self.dytpe)
        domain.data_v = torch.zeros(2, self.w, self.h).to(self.dytpe)
        domain.base_propagation = self.getPMLField().to(self.dytpe)
        domain.sourceList = self.sourceManager.getRandomSourceList()
        domain.propagation_p = None
        domain.propagation_v = None
        return domain

    def getSpecifiedDomain(self, index: int) -> Domain:
        params = self.params
        testPointNumber = params.testPointNumber
        testSourceParamsList = params.testSourceParamsList

        domain = Domain(index)
        domain.data_p = torch.zeros(1, self.w, self.h).to(self.dytpe)
        domain.data_v = torch.zeros(2, self.w, self.h).to(self.dytpe)
        domain.base_propagation = self.getPMLField().to(self.dytpe)
        domain.sourceList = []
        for sourceParams in params.testSourceParamsList:
            testT, testBias, testSourceX, testSourceY, testSourceWH = sourceParams
            source = self.sourceManager.getSourceBySet(
                testT, testBias, testSourceX, testSourceY, testSourceWH
            )
            domain.sourceList.append(source)
        domain.propagation_p = None
        domain.propagation_v = None
        return domain

    """更新计算域"""

    def updateDomain(self, domain: Domain):
        domain.step += 1
        # update propagation field
        domain.propagation_p = domain.base_propagation.clone()
        domain.propagation_v = torch.cat([domain.base_propagation] * 2, dim=0)
        for source in domain.sourceList:
            sx = source.x
            sy = source.y
            sw = source.sourceWidth
            sh = source.sourceHeight
            mask: torch.Tensor = source.mask
            # update source
            sourceExpression = source.sourceExpression
            change = sourceExpression[domain.step % len(sourceExpression)]
            domain.data_p[:, sx : sx + sw, sy : sy + sh,] += mask * (
                change
                - domain.data_p[
                    :,
                    sx : sx + sw,
                    sy : sy + sh,
                ]
            )
            domain.propagation_p[0:1, sx : sx + sw, sy : sy + sh] = (
                mask != 1.0
            ).float()
            domain.propagation_v[0:1, sx : sx + sw, sy : sy + sh] = (
                self.derivatives.mean_top(mask) != 1.0
            ).float()
            domain.propagation_v[1:2, sx : sx + sw, sy : sy + sh] = (
                self.derivatives.mean_left(mask) != 1.0
            ).float()

    """获取PML层"""

    def getPMLField(self):
        field_line = torch.flip(
            1
            - np.log(1 / (0.001))
            * (1.5 * self.params.c / self.params.pml_width)
            * (torch.arange(self.params.pml_width) / self.params.pml_width) ** 2,
            dims=[0],
        )
        field_line /= torch.max(field_line)
        xField = torch.ones(1, self.w, self.h).to(self.dytpe)
        yField = torch.ones(1, self.w, self.h).to(self.dytpe)
        xField[:, : self.params.pml_width, :] *= (field_line).unsqueeze(0).unsqueeze(2)
        xField[:, -self.params.pml_width :, :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(2)
        )
        yField[:, :, : self.params.pml_width] *= (field_line).unsqueeze(0).unsqueeze(1)
        yField[:, :, -self.params.pml_width :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(1)
        )
        return torch.min(xField, yField)


def test1():
    from matplotlib import pyplot as plt

    index: int = 0

    params = Params()
    params.domainWidth: int = 200
    params.domainHeight: int = 200
    params.whRate: float = 0.07
    params.boundaryRate: float = 0.2
    params.pointNumber = 200
    params.minT = 30
    params.maxT = 200
    params.minBiasRate = 0.3
    params.maxBiasRate = 0.7
    context = Context(params)
    domainManager = DomainManager(context)
    domain = domainManager.getDomain(index)
    domainManager.updateDomain(domain)
    plt.matshow(domain.propagation_v[0].squeeze().numpy())
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
    line = pmlField[:, :, 100].squeeze().numpy()
    plt.scatter(range(len(line)), line)
    plt.show()


if __name__ == "__main__":
    test1()

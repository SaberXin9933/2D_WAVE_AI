from Domain import Domain
import torch
import numpy as np
from SourceManager import SourceManager
from FinitDiffenceManager import FinitDiffenceManager
from Derivatives import Derivatives
from Context import Context, Params
import random


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
        self.basePML = self.getPMLField()

    """获取随机正方形传播域"""

    def getRandomField(self, t=None):
        t = t if t != None else np.random.rand() * 100000000
        w, h = self.w, self.h
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
        data /= torch.max(data)
        return data.unsqueeze(0)

    # 获取传播域
    def getPropagation(self, domain: Domain):
        propagation = torch.ones(1, self.w, self.h)
        mean_left = self.derivatives.mean_left
        mean_top = self.derivatives.mean_top

        for source in domain.sourceList:
            sx = source.x
            sy = source.y
            sw = source.sourceWidth
            sh = source.sourceHeight
            mask: torch.Tensor = source.mask
            propagation[0:1, sx : sx + sw, sy : sy + sh] = (
                mean_left(mean_top(mask)) < 0
            ).float()
        return propagation

    """获取计算域"""

    def getDomain(self, index: int, domainType: str = None) -> Domain:
        params = self.params
        if params.type == "test" and params.testIsRandom == False:
            return self.getSpecifiedDomain(index)
        else:
            if domainType == None:
                domainType = random.sample(self.params.env_types, 1)[0]
            if domainType == "super_simple":
                return self.getSimpleDomain(index, 1)
            elif domainType == "simple":
                return self.getSimpleDomain(index)
            elif domainType == "random":
                return self.getRandomDomain(index)
            else:
                raise RuntimeError

    def getSimpleDomain(self, index: int, sourceNum: int = None) -> Domain:
        domain = Domain(index)
        domain.type = "super_simple" if sourceNum == 1 else "simple"
        domain.data_p = torch.zeros(1, self.w, self.h)
        domain.data_v = torch.zeros(2, self.w, self.h)
        domain.sourceList = self.sourceManager.getRandomSourceList(sourceNum)
        domain.base_propagation = self.getPropagation(domain) * self.getPMLField()
        #
        self.addRandomPV(domain.data_p[:], domain.data_v[:])
        return domain

    def getRandomDomain(self, index: int) -> Domain:
        domain = Domain(index)
        domain.type = "random"
        domain.data_p = torch.zeros(1, self.w, self.h)
        domain.data_v = torch.zeros(2, self.w, self.h)
        domain.base_propagation = torch.zeros(1, self.w, self.h)
        domain.sourceList = None
        #
        self.addRandomPV(domain.data_p[:], domain.data_v[:],scale=1)
        return domain

    def getSpecifiedDomain(self, index: int) -> Domain:
        params = self.params
        domain = Domain(index)
        domain.type = "specified_super_simple"
        domain.data_p = torch.zeros(1, self.w, self.h)
        domain.data_v = torch.zeros(2, self.w, self.h)
        domain.base_propagation = self.getPMLField()
        domain.sourceList = []
        for sourceParams in params.testSourceParamsList:
            testT, testBias, testSourceX, testSourceY, testSourceWH = sourceParams
            source = self.sourceManager.getSourceBySet(
                testT, testBias, testSourceX, testSourceY, testSourceWH
            )
            domain.sourceList.append(source)
        return domain

    """更新计算域"""

    def updateDomainP(self, domain: Domain):
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

    def updateRandomDomain(self, domain: Domain):
        domain.data_p[:] = 0
        domain.data_v[:] = 0
        self.addRandomPV(domain.data_p[:], domain.data_v[:], scale=1)
        domain.base_propagation = (
            torch.abs(
                self.getRandomField((domain.step - 100 * np.random.rand()) * (-100))
            )
            * self.basePML
        )
        domain.data_p *= domain.base_propagation
        domain.data_v[0:1] *= domain.base_propagation
        domain.data_v[1:2] *= domain.base_propagation

    def updateDomain(self, domain: Domain):
        domain.step += 1
        # update propagation field
        if domain.type == "random":
            self.updateRandomDomain(domain)
        if domain.type in ["super_simple", "simple","specified_super_simple"]:
            self.updateDomainP(domain)

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
        xField = torch.ones(1, self.w, self.h)
        yField = torch.ones(1, self.w, self.h)
        xField[:, : self.params.pml_width, :] *= (field_line).unsqueeze(0).unsqueeze(2)
        xField[:, -self.params.pml_width :, :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(2)
        )
        yField[:, :, : self.params.pml_width] *= (field_line).unsqueeze(0).unsqueeze(1)
        yField[:, :, -self.params.pml_width :] *= (
            (torch.flip(field_line, dims=[0])).unsqueeze(0).unsqueeze(1)
        )
        return torch.min(xField, yField)

    def addRandomPV(self, p, v, scale=0.05):
        t = (np.random.rand() - 0.5) * 10000
        seed = np.random.rand()
        if np.random.rand() > 0.5:
            scale *= -1
        p[:] += (self.getRandomField(seed + (-2 * t + np.random.rand()) * 100)) * scale
        v[0:1] += (self.getRandomField(seed + (t + np.random.rand()) * 100)) * scale
        v[1:2] += (self.getRandomField(seed + (t + np.random.rand()) * (-100))) * scale


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
    domain = mananger.getDomain(0, "random")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # p
    cax1 = ax1.matshow(domain.data_p.squeeze())
    ax1.set_title("P")
    cbar1 = fig.colorbar(cax1, ax=ax1, orientation="vertical")
    # vx
    cax2 = ax2.matshow(domain.data_v[0:1].squeeze())
    ax2.set_title("VX")
    cbar2 = fig.colorbar(cax2, ax=ax2, orientation="vertical")
    # vy
    cax3 = ax3.matshow(domain.data_v[1:2].squeeze())
    ax3.set_title("VY")
    cbar3 = fig.colorbar(cax3, ax=ax3, orientation="vertical")
    # base_propagation
    cax4 = ax4.matshow(domain.base_propagation.squeeze())
    ax4.set_title("base_propagation")
    cbar4 = fig.colorbar(cax4, ax=ax4, orientation="vertical")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test3()

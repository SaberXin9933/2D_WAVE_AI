from Context import Context
from Derivatives import Derivatives
import torch
from typing import List


class FinitDiffenceManager:
    def __init__(
        self,
        context: Context,
    ):
        self.params = context.params
        self.derivative = Derivatives(
            self.params.kernel_point_number,
            self.params.kernel_order,
            self.params.kernel_delta,
            kernel_dtype=self.params.dtype,
            device=context.device,
        )

    def cf_step(self, p, v, propagation_p, propagation_v):
        c, delta_t, delta_x = self.params.c, self.params.delta_t, self.params.delta_x
        dx = self.derivative.dx
        dy = self.derivative.dy
        move_bottom = self.derivative.move_bottom
        move_right = self.derivative.move_right

        v_new = torch.zeros_like(v)
        v_new[:, 0:1] = propagation_v[:, 0:1] * (
            v[:, 0:1, :, :] - (delta_t / delta_x) * dx(move_bottom(p))[:, :, :, :]
        )
        v_new[:, 1:2] = propagation_v[:, 1:2] * (
            v[:, 1:2, :, :] - (delta_t / delta_x) * dy(move_right(p))[:, :, :, :]
        )
        p_new = propagation_p * (
            p[:, :, :, :]
            - (c**2 * delta_t / delta_x)
            * (dx(v_new[:, 0:1, :, :]) + dy(v_new[:, 1:2, :, :]))
        )
        return (p_new, v_new)

    def physic_cf_loss(self, p_old, v_old, p_new, v_new, propagation_p, propagation_v):
        c, delta_t, delta_x = self.params.c, self.params.delta_t, self.params.delta_x
        dx = self.derivative.dx
        dy = self.derivative.dy
        move_bottom = self.derivative.move_bottom
        move_right = self.derivative.move_right

        lossBatchVX = v_new[:, 0:1] - propagation_v[:, 0:1] * (
            v_old[:, 0:1, :, :]
            - (delta_t / delta_x) * dx(move_bottom(p_old))[:, :, :, :]
        )
        lossBatchVY = v_new[:, 1:2] - propagation_v[:, 1:2] * (
            v_old[:, 1:2, :, :]
            - (delta_t / delta_x) * dy(move_right(p_old))[:, :, :, :]
        )
        lossBatchP = p_new - propagation_p * (
            p_old[:, :, :, :]
            - (c**2 * delta_t / delta_x)
            * (dx(v_new[:, 0:1, :, :]) + dy(v_new[:, 1:2, :, :]))
        )

        return lossBatchP, lossBatchVX, lossBatchVY

    """
    获取源项周围点的坐标
    """

    def getPointAroundSource(self, mask: torch.Tensor) -> List[tuple]:
        mask = (mask == 0).float() * (
            self.derivative.mean_xy(mask.unsqueeze(0).unsqueeze(1)).squeeze() > 0
        ).float()
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        # 存储值为1的区域周围的点的绝对坐标
        neighbor_coordinates = set()
        row_indices, col_indices = torch.where(mask == 1)
        for x, y in zip(row_indices, col_indices):
            neighbor_coordinates.add((int(x), int(y)))

        coordinatesSortByAround = []
        row, col = neighbor_coordinates.pop()
        while row != None and col != None:
            coordinatesSortByAround.append((row, col))
            for offset_r, offset_c in neighbor_offsets:
                new_r, new_c = row + offset_r, col + offset_c
                if (new_r, new_c) in neighbor_coordinates:
                    row, col = new_r, new_c
                    neighbor_coordinates.remove((new_r, new_c))
                    break
            else:
                row, col = None, None
        return coordinatesSortByAround

    def getSourceSurroundingsValue(self, v: torch.Tensor, mask: torch.Tensor):
        mask = (mask == 0).float().squeeze()
        v = v.squeeze()
        # 获取起始点
        neighbor_coordinates_list = self.getPointAroundSource(mask.clone())
        return [v[x, y] for x, y in neighbor_coordinates_list]


"""测试代码"""


# 基础差分迭代测试
def test1():
    from DatasetsManager import DatasetsManager, Params
    from matplotlib import pyplot as plt
    from utils.pltUtils import predictPlot

    vmin = None
    vmax = None
    params = Params()
    params.kernel_point_number = 4
    params.minT = 15
    params.maxT = 40
    params.batch_size = 1
    params.datasetNum = 1
    params.dataset_size = 1
    params.is_cuda = False
    params.datasetNum = 1
    params.type = "test"
    params.testIsRandom = True
    params.env_types = ["random"]
    context = Context(params)

    finitDiffenceManager = FinitDiffenceManager(context)
    dataSets = DatasetsManager(context)
    dataSets.startAll()

    for i in range(10000):
        data = dataSets.ask()
        index_list, batchP, batchV, propagation_p, propagation_v = data
        # batchPropagation = torch.ones_like(batchPropagation)
        newP, newV = finitDiffenceManager.cf_step(
            batchP, batchV, propagation_p, propagation_v
        )
        dataSets.tell((index_list, newP, newV))
        loss_p, loss_vx, loss_vy = finitDiffenceManager.physic_cf_loss(
            batchP, batchV, newP, newV, propagation_p, propagation_v
        )
        print(
            i,
            torch.mean(loss_p**2),
            torch.mean(loss_vx**2),
            torch.mean(loss_vy**2),
        )
        if i % 50 == 0:
            predictPlot(
                batchP[:, 0:1].detach().squeeze().cpu().numpy(),
                batchV[:, 0:1].detach().squeeze().cpu().numpy(),
                batchV[:, 1:2].detach().squeeze().cpu().numpy(),
                loss_p.detach().squeeze().cpu().numpy(),
                loss_vx.detach().squeeze().cpu().numpy(),
                loss_vy.detach().squeeze().cpu().numpy(),
                0.1,
            )


# 源项周围点值获取测试
def test2():
    from DatasetsManager import DataSets, Params
    from matplotlib import pyplot as plt
    from utils.pltUtils import predictPlot, makePlotArgs, plot_figs

    vmin = None
    vmax = None
    params = Params()
    params.kernel_point_number = 4
    params.minT = 15
    params.maxT = 40
    params.batch_size = 1
    params.datasetNum = 1
    params.dataset_size = 1
    params.is_cuda = False
    params.datasetNum = 1
    params.type = "test"
    params.testIsRandom = True
    context = Context(params)

    finitDiffenceManager = FinitDiffenceManager(context)
    dataSets = DataSets(context)
    for i in range(50):
        data = dataSets.ask()
        index_list, batchP, batchV, propagation_p, propagation_v = data
        # batchPropagation = torch.ones_like(batchPropagation)
        newP, newV = finitDiffenceManager.cf_step(
            batchP, batchV, propagation_p, propagation_v
        )
        dataSets.updateData((index_list, newP, newV))
    p_around = finitDiffenceManager.getSourceSurroundingsValue(batchP, propagation_p)

    x = [i for i in range(len(p_around))]
    data1 = makePlotArgs(x, p_around, title="p around", plotType="plot")
    data2 = makePlotArgs(Z=batchP.squeeze(), title="p", plotType="matshow")
    plot_figs([data1, data2], -1)


if __name__ == "__main__":
    test1()

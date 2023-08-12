from Context import Context
from Derivatives import Derivatives
import torch


class FinitDiffenceManager:
    def __init__(self, context: Context):
        self.params = context.params
        self.derivative = Derivatives(context)

    def cf_step(self, p, v, field):
        c, delta_t, delta_x = self.params.c, self.params.delta_t, self.params.delta_x
        dx = self.derivative.dx
        dy = self.derivative.dy
        move_bottom = self.derivative.move_bottom
        move_right = self.derivative.move_right

        v_new = torch.zeros_like(v)
        v_new[:, 0:1] = field * (
            v[:, 0:1, :, :] - (delta_t / delta_x) * dx(move_bottom(p))[:, :, :, :]
        )
        v_new[:, 1:2] = field * (
            v[:, 1:2, :, :] - (delta_t / delta_x) * dy(move_right(p))[:, :, :, :]
        )
        p_new = field * (
            p[:, :, :, :]
            - (c**2 * delta_t / delta_x)
            * (dx(v_new[:, 0:1, :, :]) + dy(v_new[:, 1:2, :, :]))
        )
        return (p_new, v_new)

    def physic_cf_loss(self, p_old, v_old, p_new, v_new, field):
        c, delta_t, delta_x = self.params.c, self.params.delta_t, self.params.delta_x
        dx = self.derivative.dx
        dy = self.derivative.dy
        move_bottom = self.derivative.move_bottom
        move_right = self.derivative.move_right

        lossBatchVX = v_new[:, 0:1] - field * (
            v_old[:, 0:1, :, :]
            - (delta_t / delta_x) * dx(move_bottom(p_old))[:, :, :, :]
        )
        lossBatchVY = v_new[:, 1:2] - field * (
            v_old[:, 1:2, :, :]
            - (delta_t / delta_x) * dy(move_right(p_old))[:, :, :, :]
        )
        lossBatchP = p_new - field * (
            p_old[:, :, :, :]
            - (c**2 * delta_t / delta_x)
            * (dx(v_new[:, 0:1, :, :]) + dy(v_new[:, 1:2, :, :]))
        )

        return lossBatchP, lossBatchVX, lossBatchVY


def test1():
    from DataSets import DataSets, Params
    from matplotlib import pyplot as plt

    params = Params()
    params.kernel_point_number = 4
    params.minT = 15
    params.maxT = 40
    params.batch_size = 1
    params.dataset_size = 1
    params.is_cuda = False
    context = Context(params)

    finitDiffenceManager = FinitDiffenceManager(context)
    dataSets = DataSets(context)

    sourceExpression = dataSets.domainList[0].source.sourceExpression
    plt.scatter(range(sourceExpression.size), sourceExpression)
    plt.show()
    for i in range(10000):
        index_list, batchP, batchV, batchPropagation = dataSets.ask()
        # batchPropagation = torch.ones_like(batchPropagation)
        newP, newV = finitDiffenceManager.cf_step(batchP, batchV, batchPropagation)
        dataSets.updateData(index_list, newP, newV, batchPropagation)
        loss_p, loss_vx, loss_vy = finitDiffenceManager.physic_cf_loss(
            batchP, batchV, newP, newV, batchPropagation
        )
        print(
            i,
            torch.mean(loss_p**2),
            torch.mean(loss_vx**2),
            torch.mean(loss_vy**2),
        )
        if i > 100:
            plt.clf()
            plt.matshow(batchP.squeeze().numpy(), fignum=0, vmin=-1, vmax=1)
            plt.colorbar()
            plt.pause(0.01)


def test2():
    from DatasetsManager import DatasetsManager, Params
    from matplotlib import pyplot as plt

    params = Params()
    params.kernel_point_number = 4
    params.minT = 15
    params.maxT = 40
    params.batch_size = 1
    params.dataset_size = 1
    params.is_cuda = True
    params.datasetNum = 1
    context = Context(params)

    finitDiffenceManager = FinitDiffenceManager(context)
    dataSets = DatasetsManager(context)
    dataSets.startAll()

    for i in range(10000):
        data = dataSets.ask()
        index_list, batchP, batchV, batchPropagation = data
        # batchPropagation = torch.ones_like(batchPropagation)
        newP, newV = finitDiffenceManager.cf_step(batchP, batchV, batchPropagation)
        dataSets.tell((index_list, newP, newV, batchPropagation))
        loss_p, loss_vx, loss_vy = finitDiffenceManager.physic_cf_loss(
            batchP, batchV, newP, newV, batchPropagation
        )
        print(
            i,
            torch.mean(loss_p**2),
            torch.mean(loss_vx**2),
            torch.mean(loss_vy**2),
        )
        if i > 100:
            plt.clf()
            plt.matshow(batchP.cpu().squeeze().numpy(), fignum=0, vmin=-1, vmax=1)
            plt.colorbar()
            plt.pause(0.01)


if __name__ == "__main__":
    test2()

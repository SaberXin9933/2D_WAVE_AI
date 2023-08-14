from Context import Context
from Derivatives import Derivatives
import torch


class FinitDiffenceManager:
    def __init__(
        self,
        context: Context,
    ):
        self.params = context.params
        self.derivative = self.derivatives = Derivatives(
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


if __name__ == "__main__":
    test1()

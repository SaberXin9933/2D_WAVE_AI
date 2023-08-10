from Context import Context
from Derivatives import Derivatives
import torch


class FinitDiffence:
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


from DataSets import DataSets, Params
from matplotlib import pyplot as plt

params = Params()
params.batch_size = 1
params.dataset_size = 1
params.is_cuda = False
context = Context(params)


finitDiffence = FinitDiffence(context)
dataSets = DataSets(context)
for i in range(10000):
    index_list, batchP, batchV, batchPropagation = dataSets.ask()
    # batchPropagation = torch.ones_like(batchPropagation)
    newP, newV = finitDiffence.cf_step(batchP, batchV, batchPropagation)
    dataSets.updateData(index_list, newP, newV, batchPropagation)
    # plt.matshow(batchPropagation[:,:,1:-1,1:-1].squeeze().numpy(),fignum=0)
    # plt.colorbar()
    # plt.show()
    print(torch.mean(batchP),torch.max(batchP))
    if i > 300:
        plt.clf()
        plt.matshow(newP[:,:,1:-1,1:-1].squeeze().numpy(),fignum=0)
        plt.colorbar()
        plt.pause(0.01)

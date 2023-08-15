from Context import Context, Params
from DataSets import DataSets
from FinitDiffenceManager import FinitDiffenceManager
from ModelManager import ModelManager
import torch
import time, os, datetime
from matplotlib import pyplot as plt
from utils.pltUtils import predictPlot


def mse(loss):
    return torch.mean(torch.pow(loss, 2), dim=(1, 2, 3))


seed = datetime.datetime.now().timestamp()  # 时间戳作为随机seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(6)

params = Params()
params.type = "test"
params.is_cuda == True
params.name = "modelTest"
params.index = "20230814_225207"
params.dataset_size = 1
params.batch_size = 1
params.loadIndex = 20

context = Context(params)
log = context.logger

finitDiffenceManager = FinitDiffenceManager(context)
modelManager = ModelManager(context)

datasets = DataSets(context)
model, optimizer = modelManager.getNetAndOptimizer(params.loadIndex)


for i in range(params.n_batches_per_epoch):
    index_list, p_old, v_old, propagation_p, propagation_v = datasets.ask()
    p_new, v_new = model(torch.cat([p_old, v_old, propagation_p, propagation_v], dim=1))
    lossBatchP, lossBatchVX, lossBatchVY = finitDiffenceManager.physic_cf_loss(
        p_old, v_old, p_new, v_new, propagation_p, propagation_v
    )
    loss_p = mse(lossBatchP)
    loss_vx = mse(lossBatchVX)
    loss_vy = mse(lossBatchVY)
    loss = torch.mean(torch.log10(loss_p + loss_vx + loss_vy))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    datasets.updateData((index_list, p_new.detach().clone(), v_new.detach().clone()))

    loss = loss.detach().cpu().numpy()
    loss_vx = torch.mean(loss_vx).detach().cpu().numpy()
    loss_vy = torch.mean(loss_vy).detach().cpu().numpy()
    loss_p = torch.mean(loss_p).detach().cpu().numpy()
    msg = (
        f"index:{i};"
        + f" loss : {loss};"
        + f" loss_vx : {loss_vx};"
        + f" loss_vy : {loss_vy};"
        + f" loss_p : {loss_p};"
    )
    log.info(msg)

    # if i % 100 == 0:
    #     plt.clf()
    #     plt.matshow(p_new.detach().cpu().squeeze().numpy(),fignum=0)
    #     plt.colorbar()
    #     plt.pause(0.1)

    if (i + 1) % 100 == 0:
        predictPlot(
            p_old[0].detach().squeeze().cpu().numpy(),
            v_old[0, 0:1].detach().squeeze().cpu().numpy(),
            v_old[0, 1:2].detach().squeeze().cpu().numpy(),
            (lossBatchP[0] ** 2).detach().squeeze().cpu().numpy(),
            (lossBatchVX[0] ** 2).detach().squeeze().cpu().numpy(),
            (lossBatchVY[0] ** 2).detach().squeeze().cpu().numpy(),
            pauseTime=-1,
        )

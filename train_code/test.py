from Context import Context, Params
from DataSets import DataSets
from FinitDiffenceManager import FinitDiffenceManager
from ModelManager import ModelManager
import torch
import time, os, datetime
from matplotlib import pyplot as plt
from utils.pltUtils import predictPlot, makePlotArgs, plot_figs


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
params.name = "test"
params.index = "20230814_223644"
params.dataset_size = 1
params.batch_size = 1
params.loadIndex = 50

context = Context(params)
log = context.logger

finitDiffenceManager = FinitDiffenceManager(context)
modelManager = ModelManager(context)

datasets_model = DataSets(context)
datasets_cf = DataSets(context)
model, optimizer = modelManager.getNetAndOptimizer(params.loadIndex)

# 总迭代步数
step_count = 200
# 观察记录
gc_x, gc_y = 50, 50
model_gc_p_list = []
cf_gc_p_list = []

for i in range(step_count):
    index_list, p_old, v_old, propagation_p, propagation_v = datasets_model.ask()
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
    datasets_model.updateData(
        (index_list, p_new.detach().clone(), v_new.detach().clone())
    )
    # 观察点记录
    model_gc_p_list.append(p_old[0, 0, gc_x, gc_y])

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

    # if (i + 1) % 10 == 0:
    #     predictPlot(
    #         p_old[0].detach().squeeze().cpu().numpy(),
    #         v_old[0, 0:1].detach().squeeze().cpu().numpy(),
    #         v_old[0, 1:2].detach().squeeze().cpu().numpy(),
    #         (lossBatchP[0] ** 2).detach().squeeze().cpu().numpy(),
    #         (lossBatchVX[0] ** 2).detach().squeeze().cpu().numpy(),
    #         (lossBatchVY[0] ** 2).detach().squeeze().cpu().numpy(),
    #         pauseTime=-1,
    #     )

for i in range(step_count):
    data = datasets_cf.ask()
    index_list_cf, p_old_cf, v_old_cf, propagation_p_cf, propagation_v_cf = data
    # batchPropagation = torch.ones_like(batchPropagation)
    newP_cf, newV_cf = finitDiffenceManager.cf_step(
        p_old_cf, v_old_cf, propagation_p_cf, propagation_v_cf
    )
    datasets_cf.updateData((index_list_cf, newP_cf, newV_cf))
    cf_gc_p_list.append(p_old_cf[0, 0, gc_x, gc_y])


"""边界比较"""
# p_around_model = finitDiffenceManager.getSourceSurroundingsValue(
#     p_old.cpu().squeeze(), propagation_p.cpu().squeeze()
# )
# p_around_cf = finitDiffenceManager.getSourceSurroundingsValue(
#     p_old_cf.cpu().squeeze(), propagation_p_cf.cpu().squeeze()
# )

# x_model = [i for i in range(len(p_around_model))]
# data_model = makePlotArgs(
#     x_model, p_around_model, title="p model around", plotType="plot"
# )
# x_cf = [i for i in range(len(p_around_cf))]
# data_cf = makePlotArgs(x_cf, p_around_cf, title="p cf around ", plotType="plot")

# plot_figs([data_model, data_cf], -1)

"""观察点比较"""
x = [i for i in range(len(model_gc_p_list))]
data_model = makePlotArgs(x, model_gc_p_list, title="p cf model ", plotType="plot")
data_cf = makePlotArgs(x, cf_gc_p_list, title="p cf around ", plotType="plot")
plot_figs([data_model, data_cf], -1)

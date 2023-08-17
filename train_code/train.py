from Context import Context, Params
from DatasetsManager import DatasetsManager
from FinitDiffenceManager import FinitDiffenceManager
from ModelManager import ModelManager
import threading
import torch
import queue
import time, os, random
from utils.pltUtils import predictPlot
from utils.torchUtils import set_num_threads, resetRandomTorchSeed


def mse(loss):
    return torch.mean(torch.pow(loss, 2), dim=(1, 2, 3))

def rmse(loss):
    return torch.sqrt(mse(loss))


def train():
    set_num_threads(12)

    params = Params()
    params.type = "train"
    params.is_cuda = True
    params.name = "RMSE"

    context = Context(params)
    log = context.logger
    tbWriter = context.tbWriter
    img_dir = context.img_dir

    finitDiffenceManager = FinitDiffenceManager(context)
    modelManager = ModelManager(context)

    dataManager = DatasetsManager(context)
    dataManager.startAll()
    model, optimizer = modelManager.getNetAndOptimizer(params.loadIndex)

    for epoch in range(params.loadIndex + 1, params.n_epochs):
        # resetRandomTorchSeed()
        model.train()
        for i in range(params.n_batches_per_epoch):
            data = dataManager.ask()
            index_list, p_old, v_old, propagation = data
            p_new, v_new = model(
                torch.cat([p_old, v_old, propagation], dim=1)
            )
            lossBatchP, lossBatchVX, lossBatchVY = finitDiffenceManager.physic_cf_loss(
                p_old, v_old, p_new, v_new, propagation
            )
            loss_p = rmse(lossBatchP)
            loss_vx = rmse(lossBatchVX)
            loss_vy = rmse(lossBatchVY)
            loss = torch.mean(torch.log10(loss_p + loss_vx + loss_vy))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dataManager.tell(
                (
                    index_list,
                    p_new.detach(),
                    v_new.detach(),
                )
            )

            # 日志打印
            if i % 50 == 0:
                loss_mean = loss.detach().cpu().numpy()
                loss_vx_mean = torch.mean(loss_vx).detach().cpu().numpy()
                loss_vy_mean = torch.mean(loss_vy).detach().cpu().numpy()
                loss_p_mean = torch.mean(loss_p).detach().cpu().numpy()
                msg = (
                    f"epoch:{epoch},index:{i};"
                    + f" loss : {loss_mean};"
                    + f" loss_vx : {loss_vx_mean};"
                    + f" loss_vy : {loss_vy_mean};"
                    + f" loss_p : {loss_p_mean};"
                )
                log.info(msg)

            # tensorboard 打印
            if i % 100 == 0:
                tbWriter.add_scalar(
                    "loss", loss.item(), epoch * params.n_batches_per_epoch + i
                )
                tbWriter.add_scalar(
                    "loss_p", torch.mean(loss_p), epoch * params.n_batches_per_epoch + i
                )
                tbWriter.add_scalar(
                    "loss_vx",
                    torch.mean(loss_vx),
                    epoch * params.n_batches_per_epoch + i,
                )
                tbWriter.add_scalar(
                    "loss_vy",
                    torch.mean(loss_vy),
                    epoch * params.n_batches_per_epoch + i,
                )

            # 训练过程图片存储
            if i % params.train_save_per_sample_times == 0:
                save_index = random.choice(range(len(index_list)))
                save_path = (
                    f"{img_dir}/train_{i // params.train_save_per_sample_times}_p.png"
                )

                predictPlot(
                    p_old[save_index].detach().squeeze().cpu().numpy(),
                    v_old[save_index, 0:1].detach().squeeze().cpu().numpy(),
                    v_old[save_index, 1:2].detach().squeeze().cpu().numpy(),
                    lossBatchP[save_index].detach().squeeze().cpu().numpy(),
                    lossBatchVX[save_index].detach().squeeze().cpu().numpy(),
                    lossBatchVY[save_index].detach().squeeze().cpu().numpy(),
                    pauseTime=-2,
                    savePath=save_path,
                )
                log.info(
                    f"save train img success,save path:{img_dir}/train_{i // params.train_save_per_sample_times}_XXX.png"
                )

        # save model
        modelManager.saveNetAndOptimizer(model, optimizer, epoch)
        log.info(f"model save epoch:{epoch} success")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    train()

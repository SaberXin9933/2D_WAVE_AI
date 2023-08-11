from Context import Context, Params
from DataSets import DataSets, ask, tell
from FinitDiffenceManager import FinitDiffenceManager
from ModelManager import ModelManager
import threading
import torch
import queue
import time, os, random
from utils.pltUtils import save_2d_tensor_fig


def mse(loss):
    return torch.mean(torch.pow(loss, 2), dim=(1, 2, 3))


seed = 777  # seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
torch.set_num_threads(6)

params = Params()
params.type = "train"
params.is_cuda == True
params.name = "simpleTrain"

context = Context(params)
log = context.logger
tbWriter = context.tbWriter
img_dir = context.img_dir

finitDiffenceManager = FinitDiffenceManager(context)
modelManager = ModelManager(context)

datasets = DataSets(context)
model, optimizer = modelManager.getNetAndOptimizer(params.loadIndex)


ask_queue = queue.Queue(2)
tell_queue = queue.Queue(2)
threading_list = []
for _ in range(2):
    t = threading.Thread(target=ask, args=(datasets, ask_queue))
    threading_list.append(t)
for _ in range(2):
    t = threading.Thread(target=tell, args=(datasets, tell_queue))
    threading_list.append(t)
for t in threading_list:
    t.start()


for epoch in range(params.loadIndex + 1, params.n_epochs):
    model.train()
    for i in range(params.n_batches_per_epoch):
        index_list, p_old, v_old, batchPropagation = ask_queue.get()
        p_new, v_new = model(p_old, v_old, batchPropagation)
        lossBatchP, lossBatchVX, lossBatchVY = finitDiffenceManager.physic_cf_loss(
            p_old, v_old, p_new, v_new, batchPropagation
        )
        loss_p = mse(lossBatchP)
        loss_vx = mse(lossBatchVX)
        loss_vy = mse(lossBatchVY)
        loss = torch.mean(torch.log10(loss_p + loss_vx + loss_vy))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tell_queue.put(
            (
                index_list,
                p_new.detach().clone(),
                v_new.detach().clone(),
                batchPropagation.detach().clone(),
            )
        )

        # 日志打印
        if i % 50 == 0:
            loss = loss.detach().cpu().numpy()
            loss_vx = torch.mean(loss_vx).detach().cpu().numpy()
            loss_vy = torch.mean(loss_vy).detach().cpu().numpy()
            loss_p = torch.mean(loss_p).detach().cpu().numpy()
            msg = (
                f"epoch:{epoch},index:{i};"
                + f" loss : {loss};"
                + f" loss_vx : {loss_vx};"
                + f" loss_vy : {loss_vy};"
                + f" loss_p : {loss_p};"
            )
            log.info(msg)

        # tensorboard 打印
        if i % 100 == 0:
            tbWriter.add_scalar(
                "loss", loss.item(), epoch * params.n_batches_per_epoch + i
            )

        # 训练过程图片存储
        if i % params.train_save_per_sample_times == 0:
            save_index = random.choice(range(len(index_list)))
            save_2d_tensor_fig(
                f"{img_dir}/train_{i // params.train_save_per_sample_times}_p.png",
                p_old[save_index].squeeze().detach().cpu().numpy(),
            )
            save_2d_tensor_fig(
                f"{img_dir}/train_i{i // params.train_save_per_sample_times}_vx.png",
                v_old[save_index, 0].squeeze().detach().cpu().numpy(),
            )
            save_2d_tensor_fig(
                f"{img_dir}/train_i{i // params.train_save_per_sample_times}_vy.png",
                v_old[save_index, 1].squeeze().detach().cpu().numpy(),
            )

    # save model
    modelManager.saveNetAndOptimizer(model, optimizer, epoch)

import torch
from datetime import datetime


def resetRandomTorchSeed():
    seed = datetime.now().timestamp()  # seed必须是int，可以自行设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证


def set_num_threads(threads_num: int):
    torch.set_num_threads(threads_num)

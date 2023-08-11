import numpy as np
from matplotlib import pyplot as plt


# 预测结果图片存储
def save_2d_tensor_fig(img_path: str, figData: np.array, vmin=None, vmax=None):
    plt.clf()
    plt.matshow(figData, fignum=0, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(img_path)

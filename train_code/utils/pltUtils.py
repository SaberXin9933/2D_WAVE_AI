import numpy as np
from typing import List
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# 预测结果图片存储
def save_matshow(img_path: str, figData: np.array, vmin=None, vmax=None):
    plt.clf()
    plt.matshow(figData, fignum=0, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(img_path)


# 通用函数
def plot_figs(inputList: List, pauseTime=-1, savePath: str = ""):
    plt.clf()
    size = len(inputList)
    m = int(np.ceil(np.sqrt(size)))
    n = int(np.ceil(size / m))
    gs = gridspec.GridSpec(n, m)  # 2x2网格

    width = 4 * m
    height = 4 * n
    plt.gcf().set_size_inches(width, height)
    for index, input in enumerate(inputList):
        args, kwargs = input
        plotType = kwargs["plotType"]
        title = kwargs["title"]
        kwargs = filterKwargs(kwargs, ["title", "plotType"])
        if plotType == "plot":
            ax = plt.subplot(gs[index // m, index % m])
            ax.set_title(title)
            ax.plot(*args, **kwargs)
        elif plotType == "matshow":
            ax = plt.subplot(gs[index // m, index % m])
            ax.set_title(title)
            cax = ax.matshow(**kwargs)
            plt.colorbar(cax, ax=ax)
    plt.tight_layout()
    if pauseTime == -1:
        plt.show()
    elif pauseTime > 0:
        plt.pause(pauseTime)
    elif pauseTime == -2 and savePath != "":
        plt.savefig(savePath)
    else:
        return


# plot参数构建
def makePlotArgs(*args, **kwargs):
    return args, kwargs


# plot参数过滤
def filterKwargs(kwargs, filterKeyList: List[str]):
    return {key: value for key, value in kwargs.items() if key not in filterKeyList}


"""预测结果"""


def predictPlot(p, vx, vy, loss_p, loss_vx, loss_vy, pauseTime=-1, savePath: str = ""):
    plot_list = []
    plot_list.append(
        makePlotArgs(
            Z=p,
            vmin=-1.0,
            vmax=1.0,
            title="p",
            plotType="matshow",
            cmap="RdBu_r",
        )
    )
    plot_list.append(
        makePlotArgs(
            Z=vx,
            vmin=-1.0,
            vmax=1.0,
            title="vx",
            plotType="matshow",
            cmap="RdBu_r",
        )
    )
    plot_list.append(
        makePlotArgs(
            Z=vy,
            vmin=-1.0,
            vmax=1.0,
            title="vy",
            plotType="matshow",
            cmap="RdBu_r",
        )
    )
    plot_list.append(
        makePlotArgs(
            Z=loss_p, title="loss p", plotType="matshow", cmap="RdBu_r", origin="lower"
        )
    )
    plot_list.append(
        makePlotArgs(
            Z=loss_vx, title="loss vx", plotType="matshow", cmap="RdBu_r", origin="lower"
        )
    )
    plot_list.append(
        makePlotArgs(
            Z=loss_vy, title="loss vy", plotType="matshow", cmap="RdBu_r", origin="lower"
        )
    )
    plot_figs(plot_list, pauseTime=pauseTime, savePath=savePath)


"""
测试函数
"""


def test1():
    import random

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = x**2
    y4 = np.exp(x)
    v = np.ones((200, 200))
    img_path = f"./{ random.random()}test.eps"

    # 使用 gs[行, 列] 索引选择具体的网格位置
    data1 = makePlotArgs(x, y1, title="sin(x)", plotType="plot")
    data2 = makePlotArgs(x, y2, title="cos(x)", plotType="plot")
    data3 = makePlotArgs(x, y3, title="x^2", plotType="plot")
    data4 = makePlotArgs(Z=v, vmin=0.3, title="exp(x)", plotType="matshow")

    if random.random() > 0.5:
        plot_figs([data1, data2, data3, data4, data4], -2)
    else:
        plot_figs([data1, data4], -2)
    plt.savefig(img_path)


if __name__ == "__main__":
    test1()
    test1()

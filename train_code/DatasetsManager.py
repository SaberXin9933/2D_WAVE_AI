from DataSets import DataSets
from Context import Context, Params
from typing import Dict, List
import threading
import queue
import torch
import random
import time


def askThread(datasets: DataSets, ask_queue:queue.Queue,device):
    with torch.no_grad():
        while True:
            data = datasets.ask()
            ask_queue.put([di.to(device) if isinstance(di, torch.Tensor) else di for di in data])


def tellThread(datasets: DataSets, tell_queue: queue.Queue):
    device = torch.device("cpu")
    with torch.no_grad():
        while True:
            data = tell_queue.get()
            datasets.updateData([di.to(device) if isinstance(di, torch.Tensor) else di for di in data])

class DatasetsManager:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = context.params
        self.log = context.logger
        self.datasets = DataSets(context)
        # 多线程GPU CPU交互
        self.askQueue = queue.Queue(2)
        self.tellQueue = queue.Queue(2)
        # 多进程读写交互
        # self.processList

    def startAll(self):
        threading_list = []
        for _ in range(8):
            t = threading.Thread(target=askThread,args=(self.datasets,self.askQueue,self.context.device))
            threading_list.append(t)
        for _ in range(2):
            t = threading.Thread(target=tellThread,args=(self.datasets,self.tellQueue))
            threading_list.append(t)
        for t in threading_list:
            t.start()


    def ask(self):
        return self.askQueue.get()

    def tell(self, data):
        self.tellQueue.put(data)


"""
TEST代码
"""


def test1():
    from Context import Params
    from matplotlib import pyplot as plt
    import time
    torch.set_num_threads(12)
    params = Params()
    params.batch_size = 50
    params.dataset_size = 1000
    context = Context(params)
    context.device = torch.device("cuda:0")
    datasetManager = DatasetsManager(context)
    datasetManager.startAll()
    # time.sleep(5)
    start = time.time()
    tt = time.time()
    time_list = []
    for i in range(100):
        time_list.append(time.time()-tt)
        tt = time.time()
        t1 = time.time()
        data = datasetManager.ask()
        t2 = time.time()
        time.sleep(3600/100000)
        t3 = time.time()
        datasetManager.tell(data[:-1])
        t4 = time.time()

        print(i, t2 - t1, t4 - t3)
    print(time.time() - start)
    plt.plot([i for i in range(len(time_list))],time_list)
    plt.show()


if __name__ == "__main__":
    test1()

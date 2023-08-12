from DataSets import DataSets
from Context import Context, Params
from typing import Dict, List
from multiprocessing import Queue, Process
import threading
import queue
import torch
import random
import time


def askThread(datasets: DataSets, ask_queue: queue.Queue):
    with torch.no_grad():
        device = datasets.context.device
        while True:
            index_list, batchP, batchV, batchPropagation = datasets.ask()
            ask_queue.put(
                (
                    index_list,
                    batchP,
                    batchV,
                    batchPropagation,
                )
            )


def tellThread(datasets: DataSets, tell_queue: queue.Queue):
    with torch.no_grad():
        while True:
            index_list, batchP, batchV, batchPropagation = tell_queue.get()
            datasets.updateData(
                index_list,
                batchP,
                batchV,
                batchPropagation,
            )


def datasetsProcess(
    params: Params,
    ask_queue: Queue,
    tell_queue: Queue,
):
    context = Context(params)
    datasets = DataSets(context)
    threading_list = []
    for _ in range(1):
        t = threading.Thread(target=askThread, args=(datasets, ask_queue))
        threading_list.append(t)
    for _ in range(1):
        t = threading.Thread(target=tellThread, args=(datasets, tell_queue))
        threading_list.append(t)
    for t in threading_list:
        t.start()


class DatasetsManager:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = context.params
        self.log = context.logger
        # 多线程交互
        self.askQueue = queue.Queue()
        self.tellQueue = queue.Queue()
        self.indexQueue = queue.Queue()
        # 多进程
        self.askProcessQueueList: List[Queue] = []
        self.tellProcessQueueList: List[Queue] = []
        self.processList: List[Process] = []

        # init
        self.initQueueMap()
        self.initDatasetsProcess()

    def initQueueMap(self):
        datasetNum = self.params.datasetNum
        for _ in range(datasetNum):
            self.askProcessQueueList.append(Queue(1))
            self.tellProcessQueueList.append(Queue(1))

    def initDatasetsProcess(self):
        datasetNum = self.params.datasetNum
        for index in range(datasetNum):
            ask_queue = self.askProcessQueueList[index]
            tell_queue = self.tellProcessQueueList[index]
            ask_process = Process(
                target=datasetsProcess,
                args=(
                    self.params,
                    ask_queue,
                    tell_queue,
                ),
            )
            ask_process.daemon = True
            self.processList.append(ask_process)

    def startAll(self):
        for p in self.processList:
            p.start()
        threading_list = []
        for _ in range(1):
            t = threading.Thread(target=self.askThreading)
            threading_list.append(t)
        for _ in range(1):
            t = threading.Thread(target=self.tellThreading)
            threading_list.append(t)
        for t in threading_list:
            t.start()

    def askByIndex(self, index: int):
        askQueue: Queue = self.askProcessQueueList[index]
        data = askQueue.get()
        return data

    def tellByIndex(self, index: int, data: tuple):
        tellQueue: Queue = self.tellProcessQueueList[index]
        return tellQueue.put(data)

    def askThreading(self):
        device = self.context.device
        while True:
            for index in range(len(self.askProcessQueueList)):
                if self.askProcessQueueList[index].qsize() > 0:
                    currentIndex = index
                    break
            else:
                currentIndex = random.choice(
                    [i for i in range(len(self.askProcessQueueList))]
                )

            data = self.askByIndex(currentIndex)
            index_list, batchP, batchV, batchPropagation = data
            self.indexQueue.put(currentIndex)
            self.askQueue.put(
                (
                    index_list,
                    batchP.to(device),
                    batchV.to(device),
                    batchPropagation.to(device),
                )
            )

    def tellThreading(self):
        while True:
            data = self.tellQueue.get()
            index_list, batchP, batchV, batchPropagation = data
            currentIndex = self.indexQueue.get()
            self.tellByIndex(
                currentIndex, (index_list, batchP.cpu(), batchV.cpu(), batchPropagation.cpu())
            )

    def ask(self):
        return self.askQueue.get()

    def tell(self, data):
        self.tellQueue.put(data)


"""
TEST代码
"""


def test1():
    from Context import Params
    import time

    params = Params()
    params.datasetNum = 5
    params.dataset_size = 500
    context = Context(params)
    datasetManager = DatasetsManager(context)
    datasetManager.startAll()
    time.sleep(5)
    start = time.time()
    for i in range(100):
        t1 = time.time()
        data = datasetManager.ask()
        t2 = time.time()
        # time.sleep(0.1)
        t3 = time.time()
        datasetManager.tell(data)
        t4 = time.time()

        print(i, t2 - t1, t4 - t3)
    print(time.time() - start)


if __name__ == "__main__":
    test1()
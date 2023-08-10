from Params import Params
from typing import *
from Domain import Domain
from DomainManager import DomainManager
from SourceManager import SourceManager
import torch
import random
from threading import Lock as TLock
import time
import threading
import queue
import time
from Context import Context


class DataSets:
    def __init__(self, context: Context) -> None:
        self.params = params = context.params
        self.log = context.logger
        self.dataset_size = params.dataset_size
        self.batch_size = params.batch_size
        self.is_cuda = params.is_cuda
        self.askFailWaitTime = params.ask_fail_wait_time

        # 域管理类初始化
        self.sourceManager = SourceManager(
            context
        )
        self.domainManager = DomainManager(
            context,self.sourceManager
        )

        # 计算域数据初始化
        self.domainList = self.getDomainList(params.dataset_size)
        self.domainSize = params.dataset_size
        self.readIndex = 0
        self.writeCount = 0
        self.rLock = TLock()

    """获取单个计算域配置"""

    def getDomainList(self, domainSize: int) -> List[Domain]:
        self.log.info("domain generate start...")
        domainList = []
        for index in range(domainSize):
            domain = self.domainManager.getRandomDomain(index)
            domainList.append(domain)
        self.log.info("domain generate success !!!")
        return domainList

    """索引获取"""

    def askDomainByIndex(self, index: int) -> tuple:
        if index > self.domainSize:
            raise RuntimeError
        domain = self.domainList[index]
        domain.update()
        return domain

    def askDomainListByIndexList(self, indexList: List[int]) -> List[Domain]:
        return [self.askDomainByIndex(index) for index in indexList]

    """同步批量读"""

    def ask(self):
        while True:
            data = self.ansycAsk()
            if data == ():
                time.sleep(self.askFailWaitTime)
            else:
                return data

    """异步批量读"""

    def ansycAsk(self):
        self.rLock.acquire()
        if self.readIndex >= self.dataset_size:
            self.rLock.release()
            return ()
        start = self.readIndex
        end = min(start + self.batch_size, self.dataset_size)
        self.readIndex = end
        self.rLock.release()

        selected_domains = self.domainList[start:end]
        for domain in selected_domains:
            domain.update()

        selected_data_index = [domain.index for domain in selected_domains]
        selected_data_p = [domain.data_p for domain in selected_domains]
        selected_data_v = [domain.data_v for domain in selected_domains]
        selected_data_propagation = [
            domain.data_propagation for domain in selected_domains
        ]
        return (
            selected_data_index,
            torch.stack(selected_data_p, dim=0),
            torch.stack(selected_data_v, dim=0),
            torch.stack(selected_data_propagation, dim=0),
        )

    """批量回写"""

    def updateData(
        self,
        indexList: List[int],
        batchP: torch.Tensor,
        batchV: torch.Tensor,
        batchPropagation: torch.Tensor,
    ):
        batchP, batchV, batchPropagation = (
            batchP.cpu(),
            batchV.cpu(),
            batchPropagation.cpu(),
        )
        for i, index in enumerate(indexList):
            domain = self.domainList[index]
            domain.data_p = batchP[i]
            domain.data_v = batchV[i]
            domain.data_propagation = batchPropagation[i]

        self.rLock.acquire()
        self.writeCount += len(indexList)
        if self.writeCount == self.dataset_size:
            self.writeCount = 0
            self.readIndex = 0
            random.shuffle(self.domainList)
        self.rLock.release()


def ask(datasets: DataSets, ask_queue: queue.Queue):
    while True:
        index_list, batchP, batchV, batchPropagation = datasets.ask()
        ask_queue.put(
            (index_list, batchP.cuda(), batchV.cuda(), batchPropagation.cuda())
        )


def tell(datasets: DataSets, tell_queue: queue.Queue):
    while True:
        index_list, batchP, batchV, batchPropagation = tell_queue.get()
        datasets.updateData(
            index_list, batchP.cpu(), batchV.cpu(), batchPropagation.cpu()
        )


"""测试函数"""


def test1():
    params = Params()
    params.batch_size = 100
    params.dataset_size = 1000
    datasets = DataSets(params)
    N = 100
    wait = 0.1
    for i in range(N):
        start_time = time.time()
        index_list, batchP, batchV, batchPropagation = datasets.ask()
        batchP, batchV, batchPropagation = (
            batchP.cuda(),
            batchV.cuda(),
            batchPropagation.cuda(),
        )
        time.sleep(wait)
        batchP, batchV, batchPropagation = (
            batchP.cpu(),
            batchV.cpu(),
            batchPropagation.cpu(),
        )
        datasets.updateData(index_list, batchP, batchV, batchPropagation)
        end_time = time.time()
        print(i, f"ask time cost : {end_time - start_time} s")
    print([domain.step for domain in datasets.domainList])


def train_test():
    context = Context()
    params = context.params 
    params.batch_size = 100
    params.dataset_size = 1000
    datasets = DataSets(context)
    ask_queue = queue.Queue()
    tell_queue = queue.Queue()
    wait = 0.1

    t_list = []
    for _ in range(2):
        t = threading.Thread(target=ask, args=(datasets, ask_queue))
        t_list.append(t)
    for _ in range(2):
        t = threading.Thread(target=tell, args=(datasets, tell_queue))
        t_list.append(t)
    for t in t_list:
        t.start()

    cost_list = []
    for i in range(100):
        start_time = time.time()
        index_list, batchP, batchV, batchPropagation = ask_queue.get()
        time.sleep(wait)
        tell_queue.put((index_list, batchP, batchV, batchPropagation))
        end_time = time.time()
        cost = end_time - start_time - wait
        cost_list.append(cost)
        print(i, f"ask time cost : {cost} s")

    print(sum(cost_list) / len(cost_list))

    print([domain.step for domain in datasets.domainList])


if __name__ == "__main__":
    train_test()

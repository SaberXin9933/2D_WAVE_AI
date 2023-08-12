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
import numpy as np
import time
from Context import Context
import os


class DataSets:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = params = context.params
        self.log = context.logger
        self.dataset_size = params.dataset_size
        self.batch_size = params.batch_size
        self.is_cuda = params.is_cuda
        self.askFailWaitTime = params.ask_fail_wait_time

        # 域管理类初始化
        self.sourceManager = SourceManager(context)
        self.domainManager = DomainManager(context, self.sourceManager)

        # 计算域数据初始化
        self.domainList = self.getDomainList(params.dataset_size)
        self.indexList = [i for i in range(params.dataset_size)]
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
            if index != 0 and (10 * index) % domainSize == 0:
                self.log.info(f"domain has generate {100*index/(domainSize)}%")
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
        retryCount = 0
        while True:
            if retryCount > 1000:
                self.log.error(f"ask fail,max retry times")
                os._exit(1)
            data = self.ansycAsk()
            if data == ():
                retryCount += 1
                time.sleep(self.askFailWaitTime * retryCount)
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

        selected_domains = [
            self.domainList[index] for index in self.indexList[start:end]
        ]
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
        for i, index in enumerate(indexList):
            if self.params.type == "train" and random.random() < self.params.reset_freq:
                self.domainList[index] = self.domainManager.getRandomDomain(index)
                self.log.info(f"reset_{index}")
            else:
                domain = self.domainList[index]
                domain.data_p = batchP[i]
                domain.data_v = batchV[i]
                domain.data_propagation = batchPropagation[i]

        self.rLock.acquire()
        self.writeCount += len(indexList)
        if self.writeCount == self.dataset_size:
            self.writeCount = 0
            self.readIndex = 0
            random.shuffle(self.indexList)
        self.rLock.release()


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

    t1 = time.time()
    for i in range(100):
        index_list, batchP, batchV, batchPropagation = datasets.ask()
        datasets.updateData(index_list, batchP, batchV, batchPropagation)
    cost = time.time()-t1
    print(cost/100)


if __name__ == "__main__":
    train_test()

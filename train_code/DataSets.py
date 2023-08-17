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
from torch.utils.data import Dataset
from utils.otherUtils import split_list


class DataSets(Dataset):
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = params = context.params
        self.log = context.logger
        self.dataset_size = params.dataset_size
        self.batch_size = params.batch_size
        self.is_cuda = params.is_cuda
        self.askFailWaitTime = params.ask_fail_wait_time
        

        # 域管理类初始化
        self.domainManager = DomainManager(context)

        # 计算域数据初始化
        self.domainList = self.getDomainList(params.dataset_size)
        self.domainSize = params.dataset_size
        self.readIndex = 0
        self.writeCount = 0
        # 索引初始化
        self.indicesQueue = queue.Queue()
        self.tell_lock = TLock()
        self.tell_indices = []
        self.tell_count = 0
        self.freshIndicesQueue()
        

    def freshIndicesQueue(self,indices_list=[]):
        if indices_list == []:
            indices_list = [i for i in range(self.dataset_size)]
        indices_list = split_list(indices_list,self.batch_size)
        random.shuffle(indices_list)
        for indices in indices_list:
            self.indicesQueue.put(indices)
        
    """获取单个计算域配置"""

    def getDomainList(self, domainSize: int) -> List[Domain]:
        self.log.info("domain generate start...")
        domainList = []
        for index in range(domainSize):
            domain = self.domainManager.getDomain(index)
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
        self.domainManager.updateDomain(domain)
        return domain
    
    def __getitem__(self, index):
        domain:Domain = self.askDomainByIndex(index)
        return domain.index,domain.data_p,domain.data_v,domain.base_propagation
    
    def __len__(self):
        return self.dataset_size

    def askDomainListByIndexList(self, indexList: List[int]) -> List[Domain]:
        return [self.askDomainByIndex(index) for index in indexList]

    """同步批量读"""

    def ask(self):
        return self.ansycAsk()

    """异步批量读"""

    def ansycAsk(self):
        indices = self.indicesQueue.get()
        selected_domains = [self.domainList[index] for index in indices]
        for domain in selected_domains:
            self.domainManager.updateDomain(domain)

        selected_data_index_list = [domain.index for domain in selected_domains]
        selected_data_p = [domain.data_p for domain in selected_domains]
        selected_data_v = [domain.data_v for domain in selected_domains]
        selected_base_propagation = [
            domain.base_propagation for domain in selected_domains
        ]
        return (
            selected_data_index_list,
            torch.stack(selected_data_p, dim=0),
            torch.stack(selected_data_v, dim=0),
            torch.stack(selected_base_propagation, dim=0),
        )

    """批量回写"""

    def updateData(self, data: tuple):
        (
            indexList,
            batchP,
            batchV,
        ) = data
        for i, index in enumerate(indexList):
            domain = self.domainList[index]
            if self.params.type == "train":
                if random.random() < self.params.reset_freq or domain.step > self.params.max_step:
                    domain = self.domainManager.getDomain(index)
                    self.log.info(f"reset_{index}")
            else:
                domain: Domain = self.domainList[index]
                domain.data_p = batchP[i]
                domain.data_v = batchV[i]

        self.tell_lock.acquire()
        self.tell_indices.extend(indexList)
        rate = len(self.tell_indices)/self.dataset_size
        count = len(self.tell_indices)//self.batch_size
        if rate >= self.params.samper_percent and self.params.samper_percent < 1:
            print("reset")
            self.freshIndicesQueue(self.tell_indices[:count*self.batch_size])
            self.tell_indices = self.tell_indices[count*self.batch_size:]
        if rate == 1.0:
            self.freshIndicesQueue(self.tell_indices)
            self.tell_indices = []
        self.tell_lock.release()
        


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
        index_list, batchP, batchV = datasets.ask()
        batchP, batchV = (
            batchP.cuda(),
            batchV.cuda(),
        )
        time.sleep(wait)
        batchP, batchV = (
            batchP.cpu(),
            batchV.cpu(),
        )
        datasets.updateData(index_list, batchP, batchV)
        end_time = time.time()
        print(i, f"ask time cost : {end_time - start_time} s")
    print([domain.step for domain in datasets.domainList])


def train_test():
    context = Context()
    params = context.params
    params.batch_size = 50
    params.dataset_size = 1000
    datasets = DataSets(context)

    t1 = time.time()
    for i in range(100):
        print(i)
        index_list, batchP, batchV, propagation = datasets.ask()
        datasets.updateData((index_list, batchP, batchV))
    cost = time.time() - t1
    print(cost / 100)


if __name__ == "__main__":
    train_test()

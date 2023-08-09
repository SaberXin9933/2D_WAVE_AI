from Params import Params
from typing import *
from Domain import Domain
from DomainManager import DomainManager
from SourceManager import SourceManager
import torch
import random
from threading import Lock as TLock


class DataSets:
    def __init__(self, params: Params) -> None:
        # 参数
        self.params = params
        # 域管理类初始化
        self.sourceManager = SourceManager(params.domainWidth,params.domainHeight,params.whRate,
            params.boundaryRate,params.pointNumber,params.minT,params.maxT,params.minBiasRate,params.maxBiasRate)
        self.domainManager = DomainManager(params.domainWidth,params.domainHeight,self.sourceManager)
        # 计算域数据初始化
        self.domainList = self.getDomainList(params.dataset_size)
        self.domainSize = params.dataset_size
        # 读缓存
        self.cachePList = torch.zeros(params.cacheSize,1,params.domainWidth,params.domainHeight)
        self.cacheVList = torch.zeros(params.cacheSize,2,params.domainWidth,params.domainHeight)
        self.cachePropagationList = torch.zeros(params.cacheSize,1,params.domainWidth,params.domainHeight)
        self.cacheIndexList = [[-1 for _ in range(params.batch_size)] for _ in range(params.cacheSize)]
        self.cacheNextInsertIndex:int = -1
        self.cacheHasInsertIndex:int = -1
        self.cacheHasReadIndex:int = -1
        self.cacheHasWriteIndex:int = -1
        self.cacheLock = TLock()


    '''获取单个计算域配置'''
    def getDomainList(self,domainSize:int) -> List[Domain]:
        domainList = []
        for index in range(domainSize):
            domain = self.domainManager.getRandomDomain(index)
            domainList.append(domain)
        return domainList

    '''索引获取'''
    def askDomainByIndex(self,index:int):
        if index > self.domainSize:
            raise RuntimeError
        domain = self.domainList[index]
        domain.sourceUpdate()
        return domain.data_p,domain.data_v,domain.data_propagation

    '''刷新缓存'''
    def freshCacheByIndexList(self,cacheIndex:int,indexList:List[int])->bool:
        if len(indexList) != self.params.batch_size:
            raise RuntimeError

        cacheP = self.cachePList[cacheIndex]
        cacheV = self.cacheVList[cacheIndex]
        cachePropagation = self.cachePropagationList[cacheIndex]
        cacheIndex = self.cacheIndexList[cacheIndex]

        for i,index in enumerate(indexList):
           data_p,data_v,data_propagation = self.askDomainByIndex(index)
           cacheP[i] = data_p
           cacheV[i] = data_v
           cachePropagation[i] = data_propagation
           cacheIndex[i] = index

    '''异步缓存刷新'''
    def ansycFresh(self)->None:
        # 并发控制
        self.cacheLock.acquire()
        if self.cacheNextInsertIndex - self.cacheHasWriteIndex == self.params.cacheSize :
            self.cacheLock.release()
            return
        else:
            self.cacheHasInsertIndex += 1
        indexList = random.sample(range(self.domainSize),self.params.batch_size)

        cacheIndex = self.cacheHasInsertIndex
        self.cacheLock.release()
        # 刷新
        self.freshCacheByIndexList(cacheIndex,indexList)

    '''读缓存'''
    def askCacheData(self):
        # 并发控制
        data = None
        self.cacheLock.acquire()
        if self.cacheHasReadIndex >= self.cacheNextInsertIndex-1:

        hasInsertCount = self.params.cacheSize - self.cacheRemain
        if hasInsertCount > 0:
            cacheIndex = self.cacheNextInsertIndex - hasInsertCount
            data = self.cachePList[cacheIndex],self.cacheVList[cacheIndex],self.cachePropagationList[cacheIndex]
        self.cacheLock.release()
        return data


    '''批量回写'''
    def setBatchData(self,batchP:torch.Tensor,batchV:torch.Tensor,batchPropagation:torch.Tensor):
        indexList = self.cache_index
        for i,index in enumerate(indexList):
            domain = self.domainList[index]
            domain.data_p = batchP[i]
            domain.data_v = batchV[i]
            domain.data_propagation = batchPropagation[i]






if __name__ == "__main__":
    # params = Params()
    # dataSets = DataSets(params)
    # for _ in range(10):
    #     dataSets.getRandomBatchData()
    a = torch.Tensor([1,2])
    b = a[0].clone()
    a += 1
    print(a,b)
import torch
from Params import Params
from Random import fRandom
from Source import fSource
from Domain import Domain
from typing import *


class DataSets:
    def __init__(self,params:Params) -> None:
        self.params = params
        # 实验数据
        self.data_p = torch.zeros(params.dataset_size,1,params.width,params.height)
        self.data_v = torch.zeros(params.dataset_size,2,params.width,params.height)
        self.data_propagation = self.getPropagation()
        # 计算域信息配置
        self.domainList = self.getDomainList(params)


    '''随机传播域'''
    def getPropagation(params:Params)->torch.Tensor:
        data_propagation = torch.zeros(1,params.width,params.height)
        randFlag = fRandom.rand()*100000000
        data_propagation[:] = torch.Tensor(fRandom.generate_random_z_cond(randFlag,params.width,params.height))
        return data_propagation
    
    '''获取震源信息'''
    def getSourceList(params:Params)->torch.Tensor:
        dataSourceListNP = fSource.getRandomPointSourceList(params.dataset_size,params.dataset_size,params.minT,params.maxT,params.minBias,params.maxBias)
        return torch.Tensor(dataSourceListNP)
    
    '''获取计算域配置'''
    def getSingleDomainList(self,params:Params)->List[Domain]:
        dataSourceList = self.getSourceList(params)
        domainList= List[Domain]
        for index in range(params.dataset_size):
            source = dataSourceList[index]
            data_p = torch.zeros(1,params.width,params.height)
            data_v = torch.zeros(2,params.width,params.height)
            data_propagation = self.getPropagation(params)
            domain = Domain(index,source,0,data_p,data_v,data_propagation)
            domainList.append(domain)
        return domainList
    
    '''重制计算域名'''
    



    

        

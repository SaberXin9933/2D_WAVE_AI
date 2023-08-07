import torch
from Params import Params
from Random import fRandom
from Source import fSource
from Domain import Domain
from typing import *


class DataSets:
    def __init__(self, params: Params) -> None:
        self.params = params
        # 实验数据
        self.data_p = torch.zeros(
            params.dataset_size, 1, params.domainWidth, params.domainHeight)
        self.data_v = torch.zeros(
            params.dataset_size, 2, params.domainWidth, params.domainHeight)
        self.data_propagation = self.getPropagation()
        # 计算域信息配置
        self.domainList = self.getDomainList(params)

    '''随机传播域'''
    def getPropagation(params: Params) -> torch.Tensor:
        data_propagation = torch.zeros(
            1, params.domainWidth, params.domainHeight)
        randFlag = fRandom.rand()*100000000
        data_propagation[:] = torch.Tensor(
            fRandom.generate_random_z_cond(randFlag, params.domainWidth, params.domainHeight))
        return data_propagation

    '''获取单个计算域配置'''

    def getSingleDomain(self, params: Params, index: int) -> Domain:
        domain = Domain(index)
        domain.data_p = torch.zeros(1, params.domainWidth, params.domainHeight)
        domain.data_v = torch.zeros(2, params.domainWidth, params.domainHeight)
        domain.data_propagation = self.getPropagation(params)
        domain.source = fSource.getRandomSource(
            params.pointNumber, params.minT, params.maxT, params.minBiasRate, params.maxBiasRate)

    '''获取计算域配置'''

    def getSingleDomainList(self, params: Params) -> List[Domain]:
        dataSourceList = self.getSourceList(params)
        domainList = List[Domain]
        # for index in range(params.dataset_size):

        return domainList

    '''重制计算域名'''

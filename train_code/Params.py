import json
import sys
import os
import warnings


class Params(object):
    def __init__(self) -> None:
        '''
        实验配置
        '''
        self.precision = 2
        self.netName = ""
        self.train_env_types = ["super_simple", "oscillator", "simple"]
        self.batch_size = 5
        self.dataset_size: int = 1000
        self.lr = 0.0001
        self.lr_grad = 0.001
        self.domainWidth = 200
        self.domainHeight = 200
        self.pml_width = 30
        self.rho = 1
        self.c = 1
        self.delta_t = 1
        self.delta_x = 2

        '''
        源项
        '''
        self.pointNumber = 200
        self.minT = 20
        self.maxT = 200
        self.minBiasRate = 0.3
        self.maxBiasRate = 0.7
        self.whRate = 0.05
        self.boundaryRate = 0.3

        '''
        缓存配置
        '''
        self.cacheSize =  5




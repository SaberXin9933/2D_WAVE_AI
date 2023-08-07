import torch
import numpy
import Params
class DataSets:
    def __init__(self,params:Params.Params) -> None:
        self.params = params
        # 实验数据
        self.data_p = torch.zeros(params.dataset_size,1,params.width,params.height)
        self.data_v = torch.zeros(params.dataset_size,2,params.width,params.height)
        self.


    def getPML(params:Params.Params)->torch.Tensor:
        data_pml = torch.zeros(params.dataset_size,1,params.width,params.height)
        

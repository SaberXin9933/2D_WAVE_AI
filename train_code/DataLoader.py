import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import random,time
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


from DataSets import DataSets
from Context import Context ,Params
params = Params()
params.dataset_size = 1000
params.batch_size = 100
context = Context(params)
train_dataset = DataSets(context)

divece = context.device
train_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True,num_workers=0)

start = time.time()
for epoch in range(10):
    for idx,(indices,p_old,v_old,propagation) in enumerate(train_dataloader,0):
        p_old,v_old,propagation = p_old.to(divece),v_old.to(divece),propagation.to(divece)
        print(epoch,idx)
        train_dataset.updateData((indices,p_old.cpu(),v_old.cpu()))
print(time.time()-start)
        
from Context import Context
from model import *
from torch import nn
from torch.optim import Optimizer, Adam, SGD
from typing import Tuple


class ModelManager:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.params = context.params
        self.log = self.context.logger
        self.device = context.device

    def getNet(self) -> nn.Module:
        netName = self.params.netName
        netHiddenSize = self.params.netHiddenSize

        if netName == "Spinn_Wave_Model":
            pde_cnn = Spinn_Wave_Model(netHiddenSize).to(self.device)
        elif netName == "Conv2d_4layer":
            pde_cnn = Conv2d_4layer(netHiddenSize).to(self.device)
        elif netName == "Conv2d_6layer":
            pde_cnn = Conv2d_6layer(netHiddenSize).to(self.device)
        else:
            self.log.error("model not match!!!")
            raise RuntimeError
        return pde_cnn

    def getOptimizer(self, model: nn.Module) -> Optimizer:
        optimizerName = self.params.optimizerName

        if optimizerName == "Adam":
            optimizer = Adam(model.parameters(), lr=self.params.lr)
        elif optimizerName == "SGD":
            optimizer = SGD(model.parameters(), lr=self.params.lr)
        else:
            self.log.error("optimizer not match!!!")
            raise RuntimeError
        return optimizer

    def saveNetAndOptimizer(self, model: nn.Module, optimizer: Optimizer, index: int):
        PATH = f"{self.context.model_dir}/{index}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )
        self.log.info(f"save model success,path:{PATH}")

    def getNetAndOptimizer(self, index: int = -1) -> Tuple[nn.Module, Optimizer]:
        model = self.getNet()
        optimizer = self.getOptimizer(model)
        if index != None and index >= 0:
            PATH = f"{self.context.model_dir}/{index}.pth"
            checkpoint = torch.load(PATH)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.log.info(f"load net success,PATH:{PATH}")
        return model, optimizer

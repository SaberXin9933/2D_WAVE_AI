import torch


class Params(object):
    def __init__(self) -> None:
        """
        实验配置
        """
        self.precision = 2
        self.netName = "Spinn_Wave_Model"
        self.optimizerName = "Adam"
        self.netHiddenSize = 32
        self.train_env_types = ["super_simple", "oscillator", "simple"]
        self.batch_size = 100
        self.dataset_size: int = 1000
        self.lr = 0.0001
        self.lr_grad = 0.001
        self.domainWidth = 200
        self.domainHeight = 200
        self.pml_width = 40
        self.rho = 1
        self.c = 1
        self.delta_t = 1
        self.delta_x = 2

        """
        源项
        """
        self.pointNumber = 100
        self.minT = 20
        self.maxT = 100
        self.minBiasRate = 0.3
        self.maxBiasRate = 0.7
        self.whRate = 0.05
        self.boundaryRate = 0.3

        """
        差分卷积核
        """
        self.kernel_point_number = 2
        self.kernel_order = 1
        self.kernel_delta = 1

        """
        缓存配置
        """
        self.cacheSize = 5
        self.ask_fail_wait_time = 0.01

        """
        pytorch配置
        """
        self.is_cuda = True
        self.dtype = torch.float32

        """
        上下文配置
        """
        self.loadIndex = -1  # "-1"
        self.name = "default"
        self.type = "train"  # "train" "test"
        self.index = None  # None "20230810_215738"

        """
        训练配置
        """
        self.n_epochs = 1000
        self.n_batches_per_epoch = 10000
        self.reset_freq = 0.0003

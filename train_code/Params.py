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
        self.lr = 0.0001
        self.lr_grad = 0.001
        self.domainWidth = 200
        self.domainHeight = 200
        self.boundaryWH = 40
        self.pml_width = 40
        self.rho = 1
        self.c = 1
        self.delta_t = 1
        self.delta_x = 2

        """
        源项
        """
        self.pointNumber = 100
        self.decay_point = 20
        self.minT = 20
        self.maxT = 100
        self.minBiasRate = 0.3
        self.maxBiasRate = 0.7
        self.minSouceWH = 4
        self.maxSouceWH = 10
        self.cellWH = 20
        self.maxSourceNum = 5

        """
        差分卷积核
        """
        self.kernel_point_number = 2
        self.kernel_order = 1
        self.kernel_delta = 1

        """
        pytorch配置
        """
        self.is_cuda = False
        self.dtype = torch.float32
        self.device_num = 0

        """
        上下文配置
        """
        self.loadIndex = -1  # "-1"
        self.name = "default"
        self.type = None  # "train" "test"
        self.index = None  # None "20230810_215738"

        """
        训练配置
        """
        self.n_epochs = 1000
        self.n_batches_per_epoch = 10000
        self.reset_freq = 0.0003
        self.train_save_per_sample_times = 1000
        self.ask_fail_wait_time = 0.01
        self.batch_size = 100
        self.datasetNum = 5
        self.dataset_size: int = 600

        """
        测试配置
        """
        self.testIsRandom = False
        self.testPointNumber = 1
        # [[T,bias,sourceX,sourceY,sourceWH]]
        self.testSourceParamsList = [[20, 0.0, 95, 95, 10]]

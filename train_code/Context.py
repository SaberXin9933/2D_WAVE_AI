import sys
import logging
from logging.handlers import RotatingFileHandler
from config import *
from time import strftime
from utils.osUtils import count_subdirectories, copy_tree
from Params import Params
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.osUtils import check_path

class Context(object):
    def __init__(self, params=None):
        self.savePath = SAVE_PATH
        self.params = Params() if params == None else params
        self.init_dir()
        self.init_logger()
        self.init_device()
        self.init_tensorboard()
        self.code_flash_save()

        # XX
        self.logger.info("日志初始化完成")

    # 文件初始化
    def init_dir(self):
        path = check_path(f"{self.savePath}/{self.params.name}/")
        if self.params.index == None:
            self.params.index = strftime("%Y%m%d_%H%M%S")
        self.base_path = check_path(f"{path}/{self.params.index}/")
        self.log_dir = check_path(f"{self.base_path}/logs/")
        self.model_dir = check_path(f"{self.base_path}/models/")
        self.img_dir = check_path(f"{self.base_path}/imgs/")

    # 日志初始化
    def init_logger(self):
        if self.params.type == "train":
            logger_file_name = f"{self.log_dir}/{self.params.type}.log"
        else:
            logger_file_name = (
                f"{self.log_dir}/{self.params.type}_{strftime('%Y%m%d')}.log"
            )
        self.logger = logging.getLogger(self.params.name)
        self.formatter = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATEFMT)
        self.logger.addHandler(self.get_file_handler(logger_file_name))
        self.logger.addHandler(self.get_console_handler())
        # 设置日志的默认级别
        self.logger.setLevel(logging.DEBUG)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler = RotatingFileHandler(
            filename,
            maxBytes=LOG_MAX_SIZE,
            backupCount=LOG_BACKUP_COUNT,  # 日志路径  # 文件到达的大小  # 保留多少份
        )
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    # Tensorboard 初始化
    def init_tensorboard(self):
        if self.params.type != "train":
            return
        tensorboard_dir = check_path(f"{self.base_path}/tensorboard/")
        self.tbWriter = SummaryWriter(tensorboard_dir)

    # 设备初始化
    def init_device(self):
        if self.params.is_cuda:
            if not torch.cuda.is_available():
                self.logger.error("cuda didnt find gpu!!!,please check the device")
                raise RuntimeError
            device_num = 0
            os.environ["CUDA_VISIVLE_DEVICES"] = f"{device_num}"
            torch.cuda.set_device(device_num)
            self.device = torch.device(f"cuda:{device_num}")
        else:
            self.device = torch.device("cpu")
        self.logger.info("device init success")

    # 代码快照
    def code_flash_save(self):
        if self.params.type == "train":
            self.code_save_dir = check_path(f"{self.base_path}/code/")
            copy_tree(CODE_PATH, self.code_save_dir)
            self.logger.info("code save success!")

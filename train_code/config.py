import os
from utils.osUtils import check_path

PROJECT_PATH = check_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_PATH = check_path(PROJECT_PATH + "/train_code")
SAVE_PATH = check_path(PROJECT_PATH +"/result_save")

# 设置日志格式#和时间格式
LOG_FMT = "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_SIZE = 100*1024*1024
LOG_BACKUP_COUNT = 999


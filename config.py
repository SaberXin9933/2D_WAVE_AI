import os


class Config(object):
    def __init__(self) -> None:
        pass
        '''
        获取项目更目录
        '''
        self.project_file_path = self.check_path(os.path.dirname(os.path.abspath(__file__)))

        '''
        训练
        '''
        self.train_code_path = self.check_path(self.project_file_path + "/train_code")


        '''
        存储
        '''
        self.save_path = self.check_path(self.project_file_path + "/result_save")
        self.model_save_path = self.check_path(self.save_path + "/model")
        self.grid_save_path = self.check_path(self.save_path + "/grid")
        self.data_save_path = self.check_path(self.save_path + "/data")



    def check_path(self, path:str) -> str:
        if not path:
            raise EOFError
        if not os.path.exists(path=path):
            os.makedirs(path)
        return path
    
config = Config()

from .data_utils import DataReader, DataCooker
import os
import signal

class OptimizationData():
    def __init__(self, db_pid=None, worker_pid=None, data_path=None, sep=" "):
        self.data_path = data_path
        self.sep = sep
        self.worker_pid = worker_pid
        self.db_pid = db_pid

    def data(self):
        dr=DataReader()
        if self.data_path is not None:
            if not os.path.isfile(self.data_path):
                if self.db_pid is not None:
                    os.killpg(os.getpgid(self.worker_pid), signal.SIGTERM)
                    os.killpg(os.getpgid(self.db_pid), signal.SIGTERM)
                raise ValueError("There is no file " + str(self.data_path)+
                                 "Please provide full path to file.")
            else:
                counts = dr.read_data(self.data_path, self.sep)
        else:
            counts = dr.read_gtex_skin()
        cook = DataCooker(counts, inject_on_pred=True)
        data = cook.data("OutInjectionFC")
        return data

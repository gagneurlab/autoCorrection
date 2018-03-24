#!/usr/bin/env python

from hyperopt import fmin, tpe, hp, Trials
import sys
from .optimization_data import OptimizationData
from .optimization_model import OptimizationModel
from .hyopt import *
from .optimization_metrics import *
import json
import tempfile
import subprocess
from kopt.utils import merge_dicts
import pymongo
import os

def print_exp(exp_name):
    print("-" * 40 + "\nexp_name: " + exp_name)

DIR_ROOT, filename = os.path.split(__file__)
DIR_OUT_TRIALS = os.path.join(DIR_ROOT, "trials")
DIR_OUT_RESULTS = os.path.join(DIR_ROOT, "saved_models", "best")
os.makedirs(DIR_OUT_RESULTS, exist_ok=True)

KILL_TIMEOUT = 60 * 80


class ParamValues():
    def __init__(self, lr, q, epochs, batch):
        self.batch = batch
        self.q= q
        self.lr = lr
        self.epochs = epochs

class RunFN():
    def __init__(self, metric, hyper_params, pv,
                 data_path, sep, run_on_mongodb, start_mongodb,
                 db_name, exp_name, ip, port, max_evals, nr_of_workers):
        self.metric = metric
        self.hyper_params = hyper_params
        self.values = pv
        self.path = data_path
        self.sep = sep
        self.run_on_mongodb = run_on_mongodb
        self.start_mongodb = start_mongodb
        self.db_name = db_name
        self.exp_name = exp_name
        self.ip = ip
        self.port = port
        self.max_evals = max_evals
        self.nr_of_workers = nr_of_workers

    def __call__(self):
        m_pid=None
        w_pid=None
        if self.run_on_mongodb:
            if self.start_mongodb:
                mongodb_path = tempfile.mkdtemp()

                proc_args = ["mongod",
                             "--dbpath=%s" % mongodb_path,
                             "--noprealloc",
                             "--port="+str(self.port)]
                print("starting mongod", proc_args)
                mongodb_proc = subprocess.Popen(
                    proc_args,
                    cwd=mongodb_path,
                )
                #workers_list = []
                #if self.nr_of_workers > self.max_evals:
                #       self.nr_of_workers = copy(self.max_evals)
                #for p in range(1,self.nr_of_workers):
                proc_args_worker = ["hyperopt-mongo-worker",
                                    "--mongo="+str(self.ip)+":"+os.path.join(str(self.port),str(self.db_name)),
                                    "--poll-interval=0.1"]
                mongo_worker_proc = subprocess.Popen(
                    proc_args_worker,
                    env=merge_dicts(os.environ, {"PYTHONPATH": os.getcwd()}),
                )
                    #workers_list.append(mongo_worker_proc)

                m_pid = mongodb_proc.pid
                w_pid = mongo_worker_proc.pid
            try:
                trials = CMongoTrials(self.db_name, self.exp_name,
                                      ip=self.ip, port=self.port,
                                      kill_timeout=KILL_TIMEOUT)
            except pymongo.errors.ServerSelectionTimeoutError:
                print("No mongod process detected! Please use flag --start_mongodb or"+
                      " start mongoDB and workers. Port: " + str(self.port) +
                      " Host: " + str(self.ip) + " DB name: " + str(self.db_name))
                sys.exit(0)
        else:
            trials = Trials()
        dat = OptimizationData(m_pid, w_pid, self.path, self.sep)
        mod = OptimizationModel()
        if self.metric == "OutlierLoss":
            fn = CompileFN(self.db_name, self.exp_name,
                           data_fn=dat.data,
                           model_fn=mod.model,
                           add_eval_metrics={"outlier_loss": OutlierLoss()},
                           loss_metric="outlier_loss",  # which metric to optimize for
                           loss_metric_mode="min",  # try to maximize the metric
                           valid_split=None,  # use 20% of the training data for the validation set
                           save_model=None,  # checkpoint the best model
                           save_results=True,  # save the results as .json (in addition to mongoDB)
                           save_dir=DIR_OUT_TRIALS)
        elif self.metric == "OutlierRecall":
            fn = CompileFN(self.db_name, self.exp_name,
                           data_fn=dat.data,
                           model_fn=mod.model,
                           add_eval_metrics={"outlier_recall": OutlierRecall(theta=25, threshold=1000)},
                           loss_metric="outlier_recall",  # which metric to optimize for
                           loss_metric_mode="max",  # try to maximize the metric
                           valid_split=None,  # use 20% of the training data for the validation set
                           save_model=None,  # checkpoint the best model
                           save_results=True,  # save the results as .json (in addition to mongoDB)
                           save_dir=DIR_OUT_TRIALS)
        else:
            raise ValueError("No such metric: " + str(self.metric) +
                             " Available metrics for --use_metric are: 'OutlierLoss'(default), 'OutlierRecall'.")
        best = fmin(fn, self.hyper_params, trials=trials, algo=tpe.suggest, max_evals=self.max_evals)
        best['encoding_dim'] = self.values.q[best['encoding_dim']]
        best['batch_size'] = self.values.batch[best['batch_size']]
        best['epochs'] = self.values.epochs[best['epochs']]
        with open(os.path.join(DIR_OUT_RESULTS,self.exp_name+"_best.json"), 'wt') as f:
            json.dump(best, f)
        print("----------------------------------------------------")
        print("best_parameters: " + str(best))
        print("----------------------------------------------------")
        if self.start_mongodb:
            #for proc in workers_list:
                #proc.kill()
                #os.kill(proc.pid, signal.SIGKILL)
            mongo_worker_proc.kill()
            mongodb_proc.kill()


class Optimization():
    def __init__(self, metric="OutlierLoss",
                 data_path=None, sep=" ", run_on_mongodb=False, start_mongodb=False,
                 db_name="corrector", exp_name="exp1", ip="localhost",
                 port=22334, nr_of_trials=1, nr_of_workers=1, only_lr=False, only_epochs=False,
                 only_batch=False, only_q=False):
        self.metric = metric
        self.data_path = data_path
        self.sep = sep
        self.run_on_mongodb = run_on_mongodb
        self.start_mongodb = start_mongodb
        self.db_name = db_name
        self.exp_name = exp_name
        self.ip = ip
        self.port = port
        self.nr_of_trials = nr_of_trials
        self.nr_of_workers = nr_of_workers
        self.only_lr = only_lr
        self.only_epochs = only_epochs
        self.only_batch = only_batch
        self.only_q = only_q

    def __call__(self):
        print_exp(self.exp_name)
        if self.only_q:
            pv = ParamValues(
                lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-4)),
                q=(18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
                epochs=(250,),
                batch=(32,)
            )
        elif self.only_batch:
            pv = ParamValues(
                lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
                q=(23,),
                epochs=(250,),
                batch=(16, 32, 50, 100, 128, 200)
            )
        elif self.only_epochs:
            pv = ParamValues(
                lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-4)),
                q=(23,),
                epochs=(100, 120, 150, 170, 200, 250, 300, 400, 500),
                batch=(32,)
            )
        elif self.only_lr:
            pv = ParamValues(
                lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
                q=(23,),
                epochs=(250,),
                batch=(32,)
            )
        else:
            pv = ParamValues(
                lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
                q=(18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
                epochs=(100, 120, 150, 170, 200, 250, 300, 400, 500),
                batch=(16, 32, 50, 100, 128, 200)
            )

        hyper_params = {
            "data": {
            },
            "model": {
                "lr": pv.lr,
                "encoding_dim": hp.choice("encoding_dim", pv.q),  ##
            },
            "fit": {
                "epochs": hp.choice("epochs", pv.epochs),  #
                "batch_size": hp.choice("batch_size", pv.batch)
            }
        }

        run = RunFN(self.metric, hyper_params, pv,
                    self.data_path, self.sep, self.run_on_mongodb, self.start_mongodb,
                    self.db_name, self.exp_name, self.ip, self.port, self.nr_of_trials,
                    self.nr_of_workers)
        run()


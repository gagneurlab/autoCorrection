from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy as sp
from copy import deepcopy
import os
DIR, filename = os.path.split(__file__)

class OutlierData():
    def __init__(self, index, data_with_outliers):
        self.index = index
        self.data_with_outliers = data_with_outliers


class OutInjectionFC():
    def __init__(self, input_data, outlier_prob=10.0**-3,
                 fold=None, sample_names=None, gene_names=None,
                 counts_file = "out_file", seed = None):

        self.input_data=input_data
        self.outlier_prob=outlier_prob
        self.seed = seed
        if fold is None:
            self.log2fc = self.computeLog2foldChange()
            self.fold = self.set_fold()
        else:
            self.fold = np.full((input_data.shape[1]), fold, dtype=int)
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        print("Injecting!")

    def computeLog2foldChange(self):
        log2fc = np.log2((0.00001 + self.input_data)/(0.00001 + np.mean(self.input_data, axis=0)))
        return log2fc

    def set_fold(self):
        fc_mins = np.trunc(np.min(self.log2fc, axis=0))
        fc_maxs = np.trunc(np.max(self.log2fc, axis=0))
        fold = np.stack([fc_mins, fc_maxs])
        return fold

    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        if self.seed is not None:
            np.random.seed(self.seed)
        idx=np.random.choice((1,0), size=(np.multiply(self.input_data.shape[0],
                                                      self.input_data.shape[1])),
                                                      p=(self.outlier_prob,
                                                      1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.log2fc[i][j] >= 0:
                    fold = self.fold[0][j]-1
                else:
                    fold = self.fold[1][j]+1
                out_count = round(np.mean(self.input_data[:,j])*(2.0**fold))
                if out_count > 200000:
                    injected[i][j] = 200000
                else:
                    injected[i][j] = out_count
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)


class ProcessedData():
    def __init__(self, x, sf=None):
        self.data = x
        self.size_factor = sf


class TrainTestData():
    def __init__(self, x_train, x_test, sf_train=None, sf_test=None):
        self.train = x_train
        self.test = x_test
        self.size_factor_train = sf_train
        self.size_factor_test = sf_test


class TrainTestPreparation():
    def __init__(self, data, sf=None,
                 rescale_per_gene=False,
                 rescale_per_sample=False,
                 rescale_by_global_median=True,
                 divide_by_sf=False, test_size=0.1,
                 no_rescaling=True,
                 no_splitting=False):
        self.data = data
        self.sf = sf
        self.test_size = test_size
        self.rescale_per_gene = rescale_per_gene
        self.rescale_per_sample = rescale_per_sample
        self.rescale_by_global_median = rescale_by_global_median
        self.set_sf()
        if no_rescaling:
            self.splited_data = self.split_data(self.sf)
        else:
            if divide_by_sf:
                self.data = self.get_rescaled_by_sf()
                self.scaling_factor = self.sf
            else:
                self.scaling_factor = self.get_scaling_factor()
            if no_splitting:
                self.processed_data = self.get_processed_data()
            else:
                self.splited_data = self.split_data(self.scaling_factor)


    def get_median_factor(self, data, axis=None):
        if axis is None: # use global median
            median_factor = np.median(data)
            median_factor = np.repeat(median_factor, data.shape[1])
        elif axis==0: #factor is median per gene
            median_factor = np.median(data, axis)
            median_factor[median_factor == 0] = 1
        elif axis==1:
            median_factor = np.median(data, axis)
            median_factor[median_factor == 0] = 1
        return median_factor

    def get_size_factor(self):
        loggeom = np.mean(np.log1p(self.data), 0)
        sf = np.exp(np.median(np.log1p(self.data) - loggeom, 1))
        return sf

    def set_sf(self):
        if self.sf is None:
            self.sf = self.get_size_factor()

    def get_scaling_factor(self):
        if self.rescale_per_gene:
            median_factor = self.get_median_factor(self.data, axis=0)
        elif self.rescale_per_sample:
            median_factor = self.get_median_factor(self.data, axis=1)
            median_factor = median_factor.reshape(median_factor.shape[1], median_factor.shape[0])
            median_factor = np.repeat(median_factor, self.data.shpe[1], axis=1)
        elif self.rescale_by_global_median:
            median_factor = self.get_median_factor(self.data)
        scaling_factor = np.multiply(self.sf, median_factor+1)
        scaling_factor = np.power(scaling_factor, -1.0)
        if not scaling_factor.all():
            raise ValueError("At least some scaling factors are zeros:\n"+scaling_factor)
        return scaling_factor

    def get_rescaled_by_sf(self):
        self.data = self.data/self.sf
        return self.data

    def split_data(self, factor):
        x_train, x_test = train_test_split(self.data,
                                           random_state=False,
                                           test_size=self.test_size)
        sf_train, sf_test = train_test_split(factor,
                                       random_state=False,
                                       test_size=self.test_size)
        self.splited_data = TrainTestData(x_train, x_test,
                                          sf_train, sf_test)
        return self.splited_data

    def get_processed_data(self):
        self.processed_data = ProcessedData(self.data, self.scaling_factor)
        return self.processed_data


class DataReader():
    def __init__(self):
        pass

    def read_gtex_blood(self):
        path = os.path.join(DIR,"data", "whole_blood_gtex.tsv.gz")
        if not os.path.isfile(path):
            raise ValueError("The file " + str(path) + " does not exist.")
        self.data = self.read_data(path, sep=",")
        return self.data

    def read_gtex_skin(self):
        path = os.path.join(DIR, "data", "skin_gtex.tsv.gz")
        if not os.path.isfile(path):
            raise ValueError("The file " + str(path) + " does not exist.")
        self.data = self.read_data(path, sep=",")
        return self.data

    def read_skin_small(self):
        path = os.path.join(DIR, "data", "skin_small.tsv.gz")
        if not os.path.isfile(path):
            raise ValueError("The file " + str(path) + " does not exist.")
        self.data = self.read_data(path, sep=",")
        return self.data

    def read_gtex_several_tissues(self):
        path=os.path.join(DIR, "data", "wbl_br1_br2_bst_hrt_skn.tsv.gz")
        if not os.path.isfile(path):
            raise ValueError("The file " + str(path) + " does not exist.")
        self.data = self.read_data(path, sep=",")
        return self.data

    def read_data(self, path, sep=" "):
        data_pd = pd.read_csv(path, index_col=0,header=0, sep=sep)
        data = np.transpose(np.array(data_pd.values))
        return data

    def read_data_pd(self, path, sep=" "):
        data = pd.read_csv(path,  compression='infer', index_col=0, header=0, sep=sep)
        return data


class DataCooker():
    def __init__(self, counts, size_factors=None,
                 inject_outliers=True, inject_on_pred=True,
                 only_prediction=False, inj_method="OutInjectionFC",
                 pred_counts=None, pred_sf=None, seed = None):
        self.counts = counts
        self.inject_outliers = inject_outliers
        self.inject_outliers_on_pred = inject_on_pred
        self.only_prediction = only_prediction
        self.inj_method = inj_method
        self.seed = seed
        if size_factors is not None:
            self.sf = size_factors
        else:
            self.sf = np.ones_like(counts).astype(float)
        if pred_counts is not None:
            self.pred_counts = pred_counts
            if pred_sf is not None:
                self.pred_sf = pred_sf
            else:
                self.pred_sf = np.ones_like(counts).astype(float)
        else:
            self.pred_counts = deepcopy(counts)
            self.pred_sf = self.sf

    def inject(self, data):
        print("Using "+self.inj_method+" method!")
        if self.inj_method == "OutInjectionFC":
            injected_outliers = OutInjectionFC(data, seed=self.seed)
        else:
            raise ValueError("Please specify one of injection methods: 'OutInjectionFC', ...")
        return injected_outliers

    def get_count_data(self, counts, sf):
        count_data = TrainTestPreparation(data=counts,sf=sf,
                                          no_rescaling=False,
                                          no_splitting=True)
        return count_data

    def prepare_noisy(self, count_data):
        noisy = self.inject(count_data.processed_data.data)
        return noisy

    def data(self, inj_method="OutInjectionFC"):
        self.inj_method=inj_method
        count_data = self.get_count_data(self.counts,self.sf)
        pred_count_data = deepcopy(count_data)
        if self.inject_outliers_on_pred:
            if not np.array_equal(self.counts,self.pred_counts):
                pred_count_data = self.get_count_data(self.pred_counts,self.pred_sf)
            pred_noisy = self.prepare_noisy(pred_count_data)
            x_test = {'inp': pred_noisy.outlier_data.data_with_outliers,
                      'sf': pred_count_data.processed_data.size_factor}
            y_true_idx_test = np.stack([self.pred_counts.astype(int), pred_noisy.outlier_data.index])
        else:
            print("Preparing data!")
            x_test = {'inp': count_data.processed_data.data,
                      'sf': count_data.processed_data.size_factor}
            y_true_idx_test = None
        if not self.only_prediction:
            if self.inject_outliers:
                count_noisy = self.prepare_noisy(count_data)
                x_noisy_train = {'inp': count_noisy.outlier_data.data_with_outliers,
                                 'sf': count_data.processed_data.size_factor}
                x_train = count_data.processed_data.data
                x_noisy_valid = {'inp': count_noisy.outlier_data.data_with_outliers,
                                 'sf': count_data.processed_data.size_factor}
                x_valid = count_data.processed_data.data
            else:
                x_noisy_train = {'inp': count_data.processed_data.data,
                                 'sf': count_data.processed_data.size_factor}
                x_train = count_data.processed_data.data
                x_noisy_valid = {'inp': count_data.processed_data.data,
                                 'sf': count_data.processed_data.size_factor}
                x_valid = count_data.processed_data.data
            cooked_data = (x_noisy_train, x_train),(x_noisy_valid, x_valid), (x_test, y_true_idx_test)
        else:
            cooked_data = (None, None),(None, None), (x_test, None)
        return cooked_data


class Evaluation():
    def __init__(self, orig_data, correction):
        self.orig_data_rho = self.get_correlations(orig_data)
        self.corrected_data_rho = self.get_correlations(orig_data / correction)

    def get_correlations(self, data):
        rho, p = sp.stats.spearmanr(data, axis=1)
        return rho














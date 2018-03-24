import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import fdrcorrection
from keras import backend as K
from .losses import NB

class OutlierRecall():
    def __init__(self, theta, threshold):
        self.theta = theta
        self.threshold = threshold

    def __call__(self, y_true, pred_mean):
        counts = y_true[0]
        idx = y_true[1]
        outlier_table = self.get_outlier_table(counts.flatten(), pred_mean.flatten(), idx.flatten())
        recall = self.get_recall(outlier_table)
        self.get_recall(outlier_table)
        return recall

    def get_outlier_table(self, counts, mu, idx):
        p_vals = []
        theta = np.empty_like(counts)
        theta.fill(self.theta)
        for x_ij,disp_ij,mu_ij in zip(counts, theta, mu):
            cdf_val = sp.stats.nbinom.cdf(k=x_ij, n=disp_ij,
                                          p=disp_ij/(mu_ij+disp_ij) )
            pmf_at_x_ij = sp.stats.nbinom.pmf(k=x_ij, n=disp_ij,
                                              p=disp_ij/(mu_ij+disp_ij) )
            p_val = min(cdf_val, 1-cdf_val+pmf_at_x_ij, 0.5)*2
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)
        p_vals_fdr = fdrcorrection(p_vals, alpha=0.1)
        idx = idx.flatten()
        table_x = np.concatenate((p_vals_fdr[1].reshape(p_vals_fdr[1].shape[0],1),
                          p_vals_fdr[0].reshape(p_vals_fdr[0].shape[0],1)), axis=1)
        table = np.concatenate((table_x, idx.reshape(idx.shape[0],1)), axis=1)
        return table # pvals, pred (significance), true (idx)

    def get_recall(self, outlier_table):
        tn, fp, fn, tp = confusion_matrix(outlier_table[:self.threshold,2], outlier_table[:self.threshold,1]).ravel()
        recall = tp / (tp + fp)
        return recall


class OutlierLoss():
    def __init__(self):
        pass

    def __call__(self, y_true, pred_mean):
        counts = y_true[0].flatten()
        idx = y_true[1].flatten()
        pred_mean = pred_mean.flatten()
        nb = NB(out_idx=idx)
        loss_res = K.eval(nb.loss(counts,pred_mean))
        return loss_res

from __future__ import annotations

import datetime
import time
from timeit import default_timer as timer

import math

import pickle
import psutil

from multiprocessing import Process, Event

import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt

from .taichi_core import loglik_xlr_t_pa, loglik_xlr_t_r_known, loglik_xlr_t_r_unknown, get_loglik_marginal_tensor
import click
import os

"""
Author: Lu Cheng
Date: 22.06.2023
Changes:
- 24.07.2023
    + change "logfile" to "log_file" to have a consistent name
    + add 2 more attributes to Parameters: cb_id_arr, readID_arr
- 09.08.2023: Assign (np.arannge(int(self.min_theta), int(self.L), int(self.theta_step)) + 0.0)  to all_theta in function fixed_run(), and self.all_theta in both functions run() and __init__()
- 10.08.2023: Only process input pickle files without “tmp” in their names
"""

@click.command(name="infer_pa")
@click.option(
    '--pkl_input_file',
    type=str,
    help='input pickle file (result of prepare_input)',
    required=True
)
@click.option(
    '--output_dir',
    type=str,
    help='output directory',
    required=True
)
def infer_pa(pkl_input_file: str, output_dir: str):
    """
    INPUT:
    - pkl_input_file: file path (pickle) including information for each UTR region
    - output_dir: path to output_dir folder

    OUTPUT:
    - Pickle file including Parameters for each UTR region
    """
    # if not all([pkl_input_file, output_dir]):
    #     cli(['apainput', '--help'])
    #     sys.exit(1)

    if not (os.path.exists(pkl_input_file)):
        raise Exception("Given input file does not exists")
    else:
        if not (os.path.exists(os.path.join(output_dir, "pkl_output"))):
            os.makedirs(os.path.join(output_dir, "pkl_output"))
        np.random.seed(1)
        ## Ex: "/scratch/cs/nanopore/let23/SCAPE/apamix/data/Chr1/original_name.input.pkl" then file name is "original_name"
        filename = os.path.basename(pkl_input_file)[:-10]
        
        ## only process pickle file that was successfully completed in prepare_input()
        if ".tmp." in filename:
            raise Exception("The input file "+filename+" is incomplete. Please re-run prepare_input() on "+filename.split(".")[0]+".bam")
#         input_pkl_file = pkl_input_file
        out_pkl_file = os.path.join(output_dir, "pkl_output", filename + ".res.pkl")

        exit_event = Event()
        log_file = os.path.join(output_dir, "pkl_output", filename + "log.txt")

        ## remove result file that is respective to considering input pickle file
        if os.path.exists(out_pkl_file):
            os.remove(out_pkl_file)

        infer_with_watchdog = watch_dog(log_file, exit_event)(infer)
        infer_with_watchdog(pkl_input_file, out_pkl_file, n_max_apa=5)


def infer(pickle_input_file, pickle_output_file, **kwargs):
    """
    :param pickle_input_file: input pickle file, e.g. ChrX.input.pkl
            n objects, each object is a tuple,
            first element is gene UTR information (string), second element is preprocessed data frame (x, l, r, pa)
    :param pickle_output_file: output_dir pickle file, e.g. ChrX.res.pkl
            store the apamix result for each gene UTR, each is a Parameter object,
            gene UTR information is stored in field "gene_info_str"
    :param kwargs: other parameters for apamix
    :return: None
    """
    print(f"start inferring APA events from input pickle file = {pickle_input_file}. Output file = {pickle_input_file}")
    res_lst = []
    with open(pickle_input_file, 'rb') as fh:
        print("open file as file handler")
        while True:
            try:
                print("start each UTR region")
                start_t = timer()
                args = kwargs.copy()
                gene_info_str, df = pickle.load(fh)
                args["data"] = df
                args["gene_info_str"] = gene_info_str
                res = subsample_run(**args)
                end_t = timer()
                print(f"Done {gene_info_str} in {(end_t - start_t)/60} min.")
                res_lst.append(res)
            except EOFError:
                break

    with open(pickle_output_file, 'wb') as fh:
        for res in res_lst:
            print(f"save result of {res.gene_info_str}")
            pickle.dump(res, fh)


"""
Author: Lu Cheng, Guangzhao Cheng
Date: 10.05.2023
"""

# calculate estimated density, last component is uniform component
def est_density(para, x_arr):
    K = para.K
    y_arr = np.zeros(len(x_arr))
    for k in range(K):
        y_arr += para.ws[k] * stats.norm(loc=para.alpha_arr[k], scale=para.beta_arr[k]).pdf(x_arr)
    y_arr += para.ws[K] * 1 / para.L
    return y_arr


# plot density given by the parameters
def plot_para(para, x_arr=None, line_style='-', color=None, label=None):
    if x_arr is None:
        x_arr = np.arange(para.L + 200)
    y_arr = est_density(para, x_arr)
    alpha_inds = np.searchsorted(x_arr, para.alpha_arr)

    plt.plot(x_arr, y_arr, linestyle=line_style, label=label, color=color)
    plt.vlines(para.alpha_arr, ymin=0, ymax=y_arr[alpha_inds], linestyle=line_style, color=color)


# plot estimated result
def plot_est_vs_real(est_para, real_para):
    """
    plot the estimated result versus
    :param est_para: estimated parameters
    :param real_para: ground truth parameters
    :return:
    """
    x_arr = np.arange(est_para.L + 200)
    pred_y_arr = est_density(est_para, x_arr)
    real_y_arr = est_density(real_para, x_arr)

    plt.style.use('ggplot')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plot_para(est_para, x_arr=x_arr, line_style='--', color=colors[0], label='pred')
    plot_para(real_para, x_arr=x_arr, line_style=':', color=colors[1], label='real')
    plt.legend(loc='best')

    plt.show()


# class for all parameters
class Parameters:
    def __init__(self, title='', alpha_arr=None, beta_arr=None, ws=None, L=None, cb_id_arr=None, readID_arr=None, K=None): ## tien
        self.title = title
        self.alpha_arr = alpha_arr
        self.beta_arr = beta_arr
        self.ws = ws
        self.K = len(self.alpha_arr)
        self.L = L
        self.cb_id_arr = cb_id_arr ## tien
        self.readID_arr = readID_arr ## tien

    def __str__(self):
        outstr = '-' * 10 + f'{self.title} K={self.K}' + '-' * 10 + '\n'
        if hasattr(self, 'gene_info_str'):
            outstr += f'gene info: {self.gene_info_str}\n'
        outstr += f'K={self.K} L={self.L} Last component is uniform component.\n'
        outstr += f'alpha_arr={self.alpha_arr}\n'
        outstr += f'beta_arr={self.beta_arr}\n'
        outstr += f'ws={np.around(self.ws, decimals=2)}\n'
        if hasattr(self, 'bic'):
            outstr += f'bic={np.around(self.bic, decimals=2)}\n'
        outstr += '-' * 30 + '\n'
        return outstr


# class for holding APA data
class Data:
    def __init__(self, x_arr, l_arr, r_arr, pa_arr, cnt_arr, inds, dtype):
        """
        :param x_arr: relative start positions of the reads on a gene
        :param l_arr: UTR part length of mapped R2
        :param r_arr:
        :param pa_arr: pa site detected from a junction read, np.nan if not junction read
        :param cnt_arr: counts of the read
        :param inds: index of the data points in the original data
        :param dtype: pa_site, r_known, r_unknown
        """
        self.x_arr = x_arr
        self.l_arr = l_arr
        self.r_arr = r_arr
        self.pa_arr = pa_arr
        self.cnt_arr = cnt_arr
        self.inds = inds
        self.dtype = dtype

    def __len__(self):
        return len(self.x_arr)


def bin_data(data, x_step=5, l_step=10, r_step=10, pa_step=5):
    """
    assign similiar data items into bins and represent them using bins and counts
    Args:
        data is a pandas data frame, contains 4 columns, x, l, r, pa

    Returns: new_x_arr, new_l_arr, new_r_arr, new_pa_arr, cnt_arr, idx_arr
    new_x_arr[idx_arr] roughly equals original x_arr
    Given the labels of new_x_arr, we could use label_arr[idx_arr] to get the labels for the original data
    """

    def _my_digitize(x_arr, bins_x):
        x_arr = x_arr.copy()
        x_arr[np.isnan(x_arr)] = -1
        return np.digitize(x_arr, bins_x, right=False)
    x_arr, l_arr, r_arr, pa_arr = np.array(data["x"]), np.array(data["l"]), np.array(data["r"]), np.array(data["pa"])

    bins_x = np.arange(0, x_step + np.nanmax(x_arr), x_step)
    bins_l = np.arange(0, l_step + np.nanmax(l_arr), l_step)

    if np.isnan(np.nanmax(r_arr)):
        bins_r = np.array([0, r_step])
    else:
        bins_r = np.arange(0, r_step + np.nanmax(r_arr), r_step)
    if np.isnan(np.nanmax(pa_arr)):
        bins_pa = np.array([0, pa_step])
    else:
        bins_pa = np.arange(0, pa_step + np.nanmax(pa_arr), pa_step)

    bin_label_x = _my_digitize(x_arr, bins_x)
    bin_label_l = _my_digitize(l_arr, bins_l)
    bin_label_r = _my_digitize(r_arr, bins_r)
    bin_label_pa = _my_digitize(pa_arr, bins_pa)

    bin_label_mat = np.column_stack((bin_label_x, bin_label_l, bin_label_r, bin_label_pa))
    unique_rows, idx_arr, cnt_arr = np.unique(bin_label_mat, axis=0, return_inverse=True, return_counts=True)

    new_x_arr = np.bincount(idx_arr, x_arr) / cnt_arr
    new_l_arr = np.bincount(idx_arr, l_arr) / cnt_arr
    new_r_arr = np.bincount(idx_arr, r_arr) / cnt_arr
    new_pa_arr = np.bincount(idx_arr, pa_arr) / cnt_arr

    return new_x_arr, new_l_arr, new_r_arr, new_pa_arr, cnt_arr, idx_arr


# ats mixture model inference
# using a class
class ApaModel(object):
    def __init__(self,
                 n_max_apa=5,
                 n_min_apa=1,
                 data=None,  # data frame contain all data
                 utr_length=2000,

                 # parameters about polyA tail
                 min_LA=20,
                 max_LA=150,

                 # parameters of fragment length distribution
                 mu_f=300,
                 sigma_f=50,

                 # parameters regarding pA sites
                 min_pa_gap=100,
                 max_beta=70,
                 theta_step=9,
                 beta_step=5,

                 # limits of component weights
                 min_ws=0.05,
                 max_unif_ws=0.15,

                 # infer with known parameters
                 fixed_inference_flag=False,
                 pre_para=None,
                 output_file=None,

                 debug=False):

        self.n_max_apa = n_max_apa  # maximum number of ATS sites
        self.n_min_apa = n_min_apa  # minimum number of ATS sites

        x_arr, l_arr, r_arr, pa_arr, cnt_arr, idx_arr = \
            bin_data(data, x_step=5, l_step=10, r_step=10, pa_step=5)
        self.x_arr = x_arr
        self.l_arr = l_arr
        self.r_arr = r_arr
        self.pa_arr = pa_arr
        self.cnt_arr = cnt_arr
        self.idx_arr = idx_arr # label_arr[idx_arr] will generate label for original data
        self.cb_id_arr = np.array(data["cb_id"]) ## tien add
        self.readID_arr = np.array(data["read_id"]) ## tien add

        self.n_frag = len(cnt_arr)

        pa_site_data, r_known_data, r_unknown_data = self._proc_data(self.x_arr, self.l_arr,
                                                                     self.r_arr, self.pa_arr, self.cnt_arr)
        self.pa_site_data = pa_site_data
        self.r_known_data = r_known_data
        self.r_unknown_data = r_unknown_data

        self.L = utr_length if utr_length > 2000 else 2000  # length of UTR region
        assert all([0 <= st < utr_length for st in self.x_arr])

        # polyA tail length distribution
        self.min_LA = min_LA
        self.max_LA = max_LA

        self.s_dis_arr = np.arange(self.min_LA, self.max_LA, 10)
        self.pmf_s_dis_arr = np.repeat(1 / len(self.s_dis_arr), len(self.s_dis_arr))
        self.pmf_s_dis_arr = self.pmf_s_dis_arr / sum(self.pmf_s_dis_arr)

        # fragment size information
        self.mu_f = mu_f
        self.sigma_f = sigma_f

        # parameters regarding pA sites
        self.min_pa_gap = min_pa_gap    # minimum gap between two consecutive pa sites
        self.max_beta = max_beta        # maximum std for a pa site
        self.theta_step = theta_step

        self.min_theta = int(min(self.l_arr)) + 0.0
#         self.all_theta = np.arange(self.min_theta, self.L, self.theta_step) + 0.0
        self.all_theta = np.arange(int(self.min_theta), int(self.L), int(self.theta_step)) + 0.0 # change in 09/08/2023
        self.n_all_theta = len(self.all_theta)

        # limits of component weights
        self.min_ws = min_ws  # minimum weight of APA component
        self.max_unif_ws = max_unif_ws  # maximum weight of uniform component

        # inference with fixed parameters
        self.fixed_inference_flag = fixed_inference_flag
        self.pre_para = pre_para   # pre-specified parameters, only update the weights

        # initialization and other info
        self.beta_step_size = beta_step
        self.nround = 50
        self.unif_log_lik = None
        self.predef_beta_arr = None
        self.coverage_profile = None

        self.pos_infinite = np.finfo('f').max # float("inf")
        self.neg_infinite = np.finfo('f').min # float('-inf')

        self.loglik_xlr_t_arr = None
        self.loglik_marginal_tensor = None

        self.output_file = output_file
        # debug mode or not
        self.debug = debug
        if debug:
            self.__validate()

    def _proc_data(self, x_arr, l_arr, r_arr, polya_site, cnt_arr):
        non_pa_site_inds = np.isnan(polya_site)
        r_nan_inds = np.isnan(r_arr)
        r_unknown_inds = np.logical_and(r_nan_inds, non_pa_site_inds)
        r_known_inds = np.logical_and(np.logical_not(r_nan_inds), non_pa_site_inds)
        pa_site_inds = np.logical_not(non_pa_site_inds)

        inds = pa_site_inds
        pa_site_data = Data(x_arr[inds], l_arr[inds], r_arr[inds], polya_site[inds], cnt_arr[inds], np.where(inds)[0], 'pa_site')
        inds = r_known_inds
        r_known_data = Data(x_arr[inds], l_arr[inds], r_arr[inds], None, cnt_arr[inds], np.where(inds)[0], 'r_known')
        inds = r_unknown_inds
        r_unknown_data = Data(x_arr[inds], l_arr[inds], r_arr[inds], None, cnt_arr[inds], np.where(inds)[0], 'r_unknown')
        return pa_site_data, r_known_data, r_unknown_data

    def _get_coverage_profile(self):
        coverage_cnt = np.zeros(self.L)
        for i in range(self.n_frag):
            tmp_inds = int(self.x_arr[i]) + np.arange(int(self.l_arr[i]))
            coverage_cnt[tmp_inds] += self.cnt_arr[i]
        x_arr = np.hstack([np.arange(-100, 0), np.arange(self.L), self.L + np.arange(100)])
        y_arr = np.hstack([np.zeros(100), coverage_cnt, np.zeros(100)])
        y_arr = self.ker_smooth(y_arr, bw=self.beta_step_size*3)
        return x_arr, y_arr

    def __validate(self):
        if self.fixed_inference_flag:
            assert self.n_min_apa <= self.pre_para.K <= self.n_max_apa
            assert np.all(self.pre_para.alpha_arr >= 0)
            assert np.all(self.pre_para.alpha_arr < self.L)

        assert self.max_beta >= self.beta_step_size
        assert 0 < self.min_theta < self.L

    def cal_z_k(self, para, k, log_zmat):
        # K = len(ws) - 1  # last component is uniform component
        ws = para.ws
        alpha_arr = para.alpha_arr
        beta_arr = para.beta_arr
        log_ws_k = self.neg_infinite if ws[k]<=0.0 else np.log(ws[k])

        if k < para.K:
            alpha_ind = np.searchsorted(self.all_theta, alpha_arr[k], side='left')
            beta_ind = np.searchsorted(self.predef_beta_arr, beta_arr[k], side='left')

            log_zmat[:, k] = log_ws_k + self.loglik_marginal_tensor[alpha_ind][beta_ind]
        else:
            log_zmat[:, k] = log_ws_k + self.unif_log_lik

        return log_zmat

    def norm_z(self, log_zmat):
        Z = log_zmat - np.max(log_zmat, axis=1, keepdims=True)
        Z = np.multiply(Z, self.cnt_arr[:, np.newaxis])
        Z = np.exp(Z)
        Z = Z / np.sum(Z, axis=1, keepdims=True)
        return Z

    # maximize ws given Z
    def maximize_ws(self, Z):
        ws = np.matmul(self.cnt_arr, Z)
        ws = ws/np.sum(ws)

        if ws[-1] > self.max_unif_ws:
            ws[:-1] = (1 - self.max_unif_ws) * ws[:-1] / np.sum(ws[:-1])
            ws[-1] = self.max_unif_ws
        return ws

    def max_alpha_beta(self, para, Z, k):
        alpha_arr = para.alpha_arr
        # beta_arr = para.beta_arr

        tmp_min_theta = self.min_theta if k == 0 else alpha_arr[k - 1]
        tmp_max_theta = self.L if k == len(alpha_arr)-1 else alpha_arr[k + 1]
        mask = (self.all_theta >= tmp_min_theta) & (self.all_theta <= tmp_max_theta)
        tmp_alpha_arr = self.all_theta[mask]
        log_ws_k = self.neg_infinite if para.ws[k] <= 0.0 else np.log(para.ws[k])

        res = {}
        for alpha in tmp_alpha_arr:
            for beta in self.predef_beta_arr:
                alpha_ind = np.searchsorted(self.all_theta, alpha, side='left')
                beta_ind = np.searchsorted(self.predef_beta_arr, beta, side='left')
                res[(alpha, beta)] = np.sum((log_ws_k + self.loglik_marginal_tensor[alpha_ind][beta_ind]) * Z[:, k] * self.cnt_arr)
        return max(res, key=res.get)

    def mstep(self, para, Z, k):
        tmp_sumk = np.sum(Z[:, k])
        # avoid division by zero
        if tmp_sumk < 1e-8:
            Z[:, k] += 1e-8

        para.ws = self.maximize_ws(Z)
        para.alpha_arr[k], para.beta_arr[k] = self.max_alpha_beta(para, Z, k)
        return para

    @staticmethod
    # find the nearest value in predef_arr of val
    def find_nearest(predef_arr, val_arr):
        idxs = np.searchsorted(predef_arr, val_arr, side='left')
        ret_idxs = idxs.copy()
        for i, idx in enumerate(idxs):
            if idx == 0:
                continue
            elif idx == len(predef_arr):
                ret_idxs[i] = len(predef_arr) - 1
            elif val_arr[i] - predef_arr[idx-1] >= predef_arr[idx] - val_arr[i]:
                ret_idxs[i] = idx
            else:
                ret_idxs[i] = idx - 1
        return ret_idxs, predef_arr[ret_idxs]

    # mstep when alpha_arr and beta_arr are fixed
    def mstep_fixed(self, para, Z, k):
        # avoid division by zero
        if np.sum(Z[:, k]) < 1e-8:
            Z[:, k] += 1e-8
        para.ws = self.maximize_ws(Z)
        return para

    def elbo(self, log_zmat, Z):
        lb = self.exp_log_lik(log_zmat, Z) + np.sum(self.cnt_arr * stats.entropy(Z, axis=1))
        return lb

    # log function to avoid zero
    @staticmethod
    def log(arr: np.ndarray) -> np.ndarray:
        """calculate log for nonzero elements, return an array of the same dimension"""
        return np.log(arr, out=np.full_like(arr, np.finfo('f').min), where=(arr != 0))

    # calculate the expected log joint likelihood
    def exp_log_lik(self, log_zmat, Z):
        # return np.sum(Z[Z != 0] * log_zmat[Z != 0])
        ZZ = np.multiply(Z, self.cnt_arr[:, np.newaxis])
        return np.sum(ZZ[Z != 0] * log_zmat[Z != 0])

    # uniform component likelihood
    def lik_f0(self, log=False):
        px = 1/self.L
        pl_s = 1/self.L
        pr_s = 1/self.max_LA
        res = px * pl_s * pr_s
        if log:
            return math.log(res)
        else:
            return res

    def lik_r_s(self, data: Data, s, log=False):
        if data.dtype == 'r_unknown':
            res = np.ones_like(data.r_arr) / s
        elif data.dtype == 'r_known':
            res = (data.r_arr <= s) / s
        else:
            raise Exception(f'unknown data type: {data.dtype}, must be r_known or r_unknown.')
        if log:
            return self.log(res)
        else:
            return res

    def lik_x_st(self, data: Data, s, theta, log=False):
        if log:
            return stats.norm(loc=theta + s - self.mu_f, scale=self.sigma_f).logpdf(data.x_arr)
        else:
            return stats.norm(loc=theta + s - self.mu_f, scale=self.sigma_f).pdf(data.x_arr)

    def lik_x_st_pa(self, data: Data, theta, log=False):
        if log:
            return stats.norm(loc=0, scale=self.sigma_f).logpdf(data.pa_arr-theta)
        else:
            return stats.norm(loc=0, scale=self.sigma_f).pdf(data.pa_arr-theta)

    def lik_l_xt(self, data: Data, theta, log=False):
        utr_len_arr = theta - data.x_arr
        valid_inds = (data.l_arr <= utr_len_arr)
        res = valid_inds + 0.0
        res[valid_inds] = 1/utr_len_arr[valid_inds]
        if log:
            return self.log(res)
        else:
            return res

    def loglik_xlr_t(self, theta):
        res = np.zeros(self.n_frag)
        for data in [self.pa_site_data, self.r_known_data, self.r_unknown_data]:
            if len(data) <= 0:
                continue
            res[data.inds] = self._loglik_xlr_t(data, theta)
        return res

    def _loglik_xlr_t(self, data, theta):
        if data.dtype == 'pa_site':
            taichi_res = loglik_xlr_t_pa(data.x_arr, data.l_arr, data.pa_arr, theta, self.sigma_f)
            return taichi_res
        if data.dtype == 'r_known':
            taichi_res = loglik_xlr_t_r_known(data.x_arr, data.l_arr, data.r_arr, self.s_dis_arr, self.pmf_s_dis_arr,
                                              theta, self.mu_f, self.sigma_f)
            return taichi_res
        if data.dtype == 'r_unknown':
            taichi_res = loglik_xlr_t_r_unknown(data.x_arr, data.l_arr, data.r_arr, self.s_dis_arr, self.pmf_s_dis_arr,
                                              theta, self.mu_f, self.sigma_f)
            return taichi_res
        raise Exception(f"Unknown data type {data.dtype}")

    def loglik_marginal_lxr(self, alpha, beta):
        tmpinds = np.logical_and(self.all_theta>=alpha-3*beta, self.all_theta<=alpha+3*beta)
        tmpinds = np.where(tmpinds)[0]
        tmp_theta_arr = self.all_theta[tmpinds]
        res = np.zeros((self.n_frag, len(tmp_theta_arr))) + self.neg_infinite
        logp_theta = stats.norm(loc=alpha, scale=beta).logpdf(tmp_theta_arr)
        for i, theta in enumerate(tmp_theta_arr):
            tid = tmpinds[i]
            res[:, i] = self.loglik_xlr_t_arr[:, tid] + logp_theta[i]
        return logsumexp(res, axis=1) - logsumexp(logp_theta)

    # generate random k such that each K is a group and no consecutive elements are the same
    @staticmethod
    def gen_k_arr(K, n):
        def _gen(K):
            ii = 0
            last_ind = -1
            arr = np.random.permutation(K)
            while True:
                if ii % K == 0:
                    np.random.shuffle(arr)
                    if arr[0] == last_ind:
                        tmpi = np.random.choice(K - 1) + 1
                        arr[0], arr[tmpi] = arr[tmpi], arr[0]
                    ii = 0
                    last_ind == arr[-1]
                yield arr[ii]
                ii += 1

        if K == 0 or K == 1:
            return np.zeros(n, dtype='int')
        ite = _gen(K)
        res = []
        for _ in range(n):
            res.append(next(ite))
        return np.array(res, dtype='int')

    # perform kernel smoothing of a given density array
    @staticmethod
    def ker_smooth(y_arr, bw=1.0):
        res = np.full_like(y_arr, 0)
        ny = len(y_arr)
        w_arr = -np.arange(-3 * bw, 3 * bw + 1) ** 2 / (2 * bw * bw)
        w_arr = np.exp(w_arr)  # Gaussian smoothing
        w_arr_sum = np.sum(w_arr)

        win_size = int(3 * bw)
        for i in range(win_size):
            st, en = i - win_size, i + win_size
            tmpinds = np.arange(st, en + 1) >= 0
            res[i] = np.sum(w_arr[tmpinds] * y_arr[0:en + 1]) / np.sum(w_arr[tmpinds])
        for i in range(win_size, ny - win_size):
            st, en = i - win_size, i + win_size
            res[i] = np.sum(w_arr * y_arr[st:en + 1]) / w_arr_sum
        for i in range(ny - win_size, ny):
            st, en = i - win_size, i + win_size
            tmpinds = np.arange(st, en + 1) < ny
            res[i] = np.sum(w_arr[tmpinds] * y_arr[st:ny]) / np.sum(w_arr[tmpinds])
        return res

    def cal_bic(self, log_zmat, Z):
        N, K = Z.shape
        K = K - 1
        res = -2 * self.exp_log_lik(log_zmat, Z) + (3 * K + 1) * np.log(N)  # the smaller bic, the better model
        return res

    def fixed_inference(self, para):
        para.ws = self.init_ws(len(para.alpha_arr))
        res = self.em_algo(para, fixed_inference_flag=True)
        return res

    # perform inference for K components
    def em_algo(self, para, fixed_inference_flag=False):
        lb = self.neg_infinite
        lb_arr = []
        N = self.n_frag
        K = para.K

        k_arr = self.gen_k_arr(K, self.nround)

        log_zmat = np.zeros((N, K + 1))
        for k in range(K + 1):
            log_zmat = self.cal_z_k(para, k, log_zmat)

        for i in range(self.nround):
            if self.debug:
                print('iteration=', i + 1, '  lb=', lb)

            # E-Step
            log_zmat = self.cal_z_k(para, k_arr[i], log_zmat)

            Z = self.norm_z(log_zmat)

            if fixed_inference_flag:
                para = self.mstep_fixed(para, Z, k_arr[i])
            else:
                para = self.mstep(para, Z, k_arr[i])

            lb_new = self.elbo(log_zmat, Z)
            lb_arr.append(lb_new)

            if np.abs(lb_new - lb) < np.abs(1e-6 * lb):
                break
            else:
                lb = lb_new

        if self.debug:
            if i == self.nround:
                print(f'Run all {i + 1} iterations. lb={lb}')
            else:
                print(f'Converge in  {i + 1} iterations. lb={lb}')

        bic = self.cal_bic(log_zmat, Z)
        if self.debug:
            print("bic=", bic)
            print('estimated ws:  ', np.around(para.ws, decimals=2))
            print("estimated alpha: ", np.around(para.alpha_arr, decimals=2))
            print("estimated beta: ", np.around(para.beta_arr, decimals=2))

        if self.debug:
            nd = len(lb_arr)
            if nd >= 3:
                plt.plot(list(range(nd - 3)), lb_arr[3:nd])
                plt.show()

        # sorted_inds = sorted(range(len(alpha_arr)), key=lambda k: alpha_arr[k])
        sorted_inds = np.argsort(para.alpha_arr)
        para.alpha_arr = para.alpha_arr[sorted_inds]
        para.alpha_arr = np.rint(para.alpha_arr).astype('int')  # round to nearest integer
        para.beta_arr = para.beta_arr[sorted_inds]
        para.ws[0:K] = para.ws[sorted_inds]

        if not fixed_inference_flag:
            para.title = 'Estimated parameters'
        para.bic = bic
        para.lb_arr = lb_arr

        return para

    def sample_alpha(self, n_apa):
        x_arr, y_arr = self.coverage_profile

        peak_inds, _ = find_peaks(y_arr, distance=self.min_pa_gap)
        peaks = x_arr[peak_inds]
        n_peak = len(peak_inds)

        bw = self.beta_step_size * 3
        left_inds = peak_inds - bw
        right_inds = peak_inds + bw + 1
        peaks_ws = np.zeros(n_peak)
        for i in range(n_peak):
            peaks_ws[i] = sum(y_arr[left_inds[i]:right_inds[i]])
        peaks_ws = peaks_ws / sum(peaks_ws)

        if n_apa <= n_peak:
            res = np.random.choice(peaks, size=n_apa, replace=False, p=peaks_ws)
        else:
            res = np.random.choice(self.L, size=n_apa-n_peak, replace=False)
            res = np.concatenate((peaks, res))

        shift = np.rint(5 * self.beta_step_size * (2*np.random.uniform(n_apa)-1))
        res = res + shift
        res = np.sort(res)
        _, res = self.find_nearest(self.all_theta, res)

        return res

    def init_ws(self, n_apa):
        ws = np.random.uniform(size=(n_apa + 1))
        ws = ws / sum(ws)
        if ws[-1] > self.max_unif_ws:
            ws[:-1] = ws[:-1] * (1 - self.max_unif_ws)
            ws[-1] = self.max_unif_ws
        return ws

    def init_para(self, n_apa):
        alpha_arr = self.sample_alpha(n_apa)
        beta_arr = np.random.choice(self.predef_beta_arr, size=n_apa, replace=True)
        ws = self.init_ws(n_apa)

        para = Parameters(title='Initial parameters', alpha_arr=alpha_arr, beta_arr=beta_arr, ws=ws, L=self.L
                          , cb_id_arr=self.cb_id_arr ## tien
                          , readID_arr=self.readID_arr ## tien
                         )
        if self.debug:
            print(para)

        return para

    # remove components with weight less than min_ws
    def rm_component(self, para):
        rm_inds = [i for i in range(para.K) if para.ws[i] < self.min_ws]
        if len(rm_inds) == 0:
            return para

        print(f'Remove components {rm_inds} with weight less than min_ws={self.min_ws}.')
        keep_inds = np.array([i for i in range(para.K) if not para.ws[i] < self.min_ws])
        para.alpha_arr = para.alpha_arr[keep_inds]
        para.beta_arr = para.beta_arr[keep_inds]
        para.K = len(keep_inds)
        para.ws = None
        para = self.fixed_inference(para)
        return para

    def em_optim0(self, n_apa):
        n_trial = 10
        lb_arr = np.full(n_trial, self.neg_infinite)
        bic_arr = np.full(n_trial, self.pos_infinite)
        res_list = list()

        for i in range(n_trial):
            if self.debug:
                print('-----------------K=', n_apa, ' | ', 'i_trial=', i + 1, ' | n_trial=', n_trial, ' -------------')
            para = self.init_para(n_apa)

            if self.debug:
                print(para)

            res_list.append(self.em_algo(para))

            lb_arr[i] = res_list[i].lb_arr[-1]
            bic_arr[i] = res_list[i].bic

        min_ind = np.argmin(bic_arr)
        res = res_list[min_ind]

        res.title = 'Estimated Parameters'
        print(res)

        return res

    def get_label(self, para):
        N = self.n_frag
        K = para.K
        log_zmat = np.zeros((N, K + 1), dtype='float')
        for k in range(K + 1):
            log_zmat = self.cal_z_k(para, k, log_zmat)
        Z = self.norm_z(log_zmat)
        label_arr = np.argmax(Z, axis=1)
        return label_arr

    def fixed_run(self, rm_comp_flag=False):
        assert self.pre_para is not None
#         all_theta = np.arange(self.min_theta, self.L, self.theta_step)
        all_theta = np.arange(int(self.min_theta), int(self.L), int(self.theta_step)) + 0.0 ## change 09/08/2023

        max_beta = np.max(self.pre_para.beta_arr)
        min_beta = np.min(self.pre_para.beta_arr)
        theta_list = []
        for alpha in self.pre_para.alpha_arr:
            tmpinds, _ = self.find_nearest(self.all_theta, np.array([alpha - 3 * max_beta, alpha + 3 * max_beta]))
            tmp_theta_arr = all_theta[tmpinds[0]:tmpinds[1]]
            theta_list.append(tmp_theta_arr)
        self.all_theta = np.unique(np.concatenate(theta_list))
        self.predef_beta_arr = np.arange(min_beta, max_beta+self.beta_step_size, self.beta_step_size) + 0.0
        self.unif_log_lik = self.lik_f0(log=True)
        start_t = timer()
        loglik_xlr_t_arr = np.zeros((self.n_frag, len(self.all_theta)))
        for i, theta in enumerate(self.all_theta):
            loglik_xlr_t_arr[:, i] = self.loglik_xlr_t(theta)
        self.loglik_xlr_t_arr = loglik_xlr_t_arr
        self.coverage_profile = self._get_coverage_profile()
        end_t = timer()
        if self.debug:
            print(f'outer theta computation {end_t - start_t} seconds.')

        start_t = timer()

        loglik_marginal_tensor = np.zeros((len(self.all_theta), len(self.predef_beta_arr), self.n_frag))
        for i, alpha in enumerate(self.all_theta):
            for j, beta in enumerate(self.predef_beta_arr):
                loglik_marginal_tensor[i][j] = self.loglik_marginal_lxr(alpha, beta)
        self.loglik_marginal_tensor = loglik_marginal_tensor

        end_t = timer()
        if self.debug:
            print(f'outer marginal computation {end_t - start_t} seconds.')

        res = self.em_optim0(self.pre_para.K)
        if rm_comp_flag:
            res = self.rm_component(res)
        res.label_arr = self.get_label(res)[self.idx_arr]

        res.title = f'Final Result (subsample run)'
        print(res)

        return res

    def run(self, skip_lik_comp_flag=False):
        if self.n_min_apa > self.n_max_apa:
            raise Exception("n_min_apa=" + str(self.n_min_apa) + " n_max_apa=" + str(self.n_max_apa) +
                            ", n_max_apa has to be greater than n_min_apa!")

        if self.max_beta < self.beta_step_size:
            raise Exception("max_beta=" + str(self.max_beta) + " beta_step_size=" + str(self.beta_step_size) +
                            ", max_beta has to be greater than beta_step_size!")

#         self.all_theta = np.arange(self.min_theta, self.L, self.theta_step)
        self.all_theta = np.arange(int(self.min_theta), int(self.L), int(self.theta_step)) + 0.0 ## change 09/08/2023

        self.predef_beta_arr = np.arange(self.beta_step_size, self.max_beta, self.beta_step_size) + 0.0

        n_apa_trial = self.n_max_apa - self.n_min_apa + 1
        bic_arr = np.full(n_apa_trial, self.pos_infinite)
        res_list = list()

        self.unif_log_lik = self.lik_f0(log=True)

        if not skip_lik_comp_flag:
            self.coverage_profile = self._get_coverage_profile()

            start_t = timer()
            loglik_xlr_t_arr = np.zeros((self.n_frag, len(self.all_theta)))
            for i, theta in enumerate(self.all_theta):
                loglik_xlr_t_arr[:, i] = self.loglik_xlr_t(theta)
            self.loglik_xlr_t_arr = loglik_xlr_t_arr

            self.loglik_marginal_tensor = get_loglik_marginal_tensor(self.all_theta, self.predef_beta_arr, loglik_xlr_t_arr)

            end_t = timer()
            if self.debug:
                print(f'Lik point and marginal computation {end_t - start_t} seconds.')

        for i, n_apa in enumerate(range(self.n_max_apa, self.n_min_apa - 1, -1)):
            # print()
            # print(20 * '*' + ' k = ' + str(n_apa) + ' ' + 20 * '*')
            res = self.em_optim0(n_apa)
            res_list.append(res)
            bic_arr[i] = res.bic

        min_ind = np.argmin(bic_arr)
        res = res_list[min_ind]

        res = self.rm_component(res)
        res.label_arr = self.get_label(res)[self.idx_arr]

        res.title = f'Final Result'
        print(res)

        return res


def subsample_run(return_model=False, re_run_mode=True, gene_info_str="None", **kwargs):
    """
    bin reads and infer pa sites
    :param return_model: if ApaModel object will be returned
    :param re_run_mode: if re-run if the infered number of pa sites equals n_max_apa
    :param gene_info_str: gene-utr_file name of the data to be analyzed
    :param kwargs: other parameters for ApaModel
    :return: a Parameter object, which contains an additional field "gene_info_str"
    """
    if not "utr_length" in kwargs:
        tbl = kwargs['data']
        utr_len = max(tbl["x"]) + max(tbl["l"]) + 50
        kwargs["utr_length"] = utr_len

    apamodel = ApaModel(**kwargs)
    res = apamodel.run()
    res.gene_info_str = gene_info_str

    while re_run_mode and len(res.alpha_arr) == kwargs["n_max_apa"]:
        print(
            f"Infer n_pa_sites = n_max_apa = {len(res.alpha_arr)}, rerun by setting n_max_apa={len(res.alpha_arr) + 2}")
        apamodel.n_max_apa = kwargs["n_max_apa"] + 2
        apamodel.n_min_apa = kwargs["n_max_apa"]
        kwargs["n_max_apa"] = kwargs["n_max_apa"] + 2
        res = apamodel.run(skip_lik_comp_flag=True)
        res.gene_info_str = gene_info_str

    if return_model:
        return res, apamodel
    else:
        return res

## Try
def exp_pa_len(apamix_res, label_arr):
    if apamix_res.K == 1:
        return 1.0
    if len(label_arr) == 0:
        return np.nan
    tmpinds = label_arr < apamix_res.K
    if not np.any(tmpinds):
        return np.nan
    uniq_labs, uniq_cnt = np.unique(label_arr[tmpinds], return_counts=True)
    ws = np.zeros(apamix_res.K)
    ws[uniq_labs] = uniq_cnt
    ws = ws/np.sum(ws)
    a_arr = apamix_res.alpha_arr
    n_arr = 1.0 + 9.0 * (a_arr - a_arr[0])/(a_arr[-1]-a_arr[0])
    return np.sum(ws*n_arr)


def cal_exp_pa_len_by_cluster(apamix_res, partition):
    partition = np.array(partition)
    uni_clusters = np.unique(partition)
    avg_len_arr = np.zeros(len(uni_clusters))
    label_arr = apamix_res.label_arr
    for i, c in enumerate(uni_clusters):
        tmplabel = label_arr[partition == c]
        avg_len_arr[i] = exp_pa_len(apamix_res, tmplabel)
    return uni_clusters, avg_len_arr


def _watch_dog(log_file: str, exit_event: Event):
    with open(log_file, "w") as fh:
        while True:
            if exit_event.is_set():
                break
            memory = psutil.virtual_memory()
            used = round(memory.used / 1024.0 / 1024.0 / 1024.0, 2)
            available = round(memory.available / 1024.0 / 1024.0 / 1024.0, 2)
            total = round(memory.total / 1024.0 / 1024.0 / 1024.0, 2)
            used_pert = round(used/total * 100, 2)
            free_pert = round(available/total * 100, 2)
            mem_info = f"Memory usage: used = {used} GB ({used_pert}%);  available={available} GB ({free_pert}%); total={total} GB"
            fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
            fh.write(f'The CPU usage is: {psutil.cpu_percent(4)}%\n')
            fh.write(f'{mem_info}\n')
            fh.write(str(memory))
            fh.write("\n\n")
            time.sleep(60)


def watch_dog(log_file: str, exit_event: Event):
    def task_func_with_watchdog(task_func):
        def wrapper_function(*args, **kwargs):
            print(f"Launching watch dog. log_file = {log_file}")
            log_proc = Process(target=_watch_dog, args=(log_file, exit_event))
            log_proc.start()
            start_t = timer()
            result = task_func(*args, **kwargs)
            end_t = timer()
            print(f"Task takes {(end_t - start_t) / 60} minutes.")
            print(f"Task finished, terminating watch dog process.")
            exit_event.set()
            log_proc.join()
            return result
        return wrapper_function
    return task_func_with_watchdog


def infer(pickle_input_file, pickle_output_file, **kwargs):
    """
    :param pickle_input_file: input pickle file, e.g. ChrX.input.pkl
            n objects, each object is a tuple,
            first element is gene UTR information (string), second element is preprocessed data frame (x, l, r, pa)
    :param pickle_output_file: output_dir pickle file, e.g. ChrX.res.pkl
            store the apamix result for each gene UTR, each is a Parameter object,
            gene UTR information is stored in field "gene_info_str"
    :param kwargs: other parameters for apamix
    :return: None
    """
    print(f"start inferring APA events from input pickle file = {pickle_input_file}. Output file = {pickle_input_file}")
    res_lst = []
    with open(pickle_input_file, 'rb') as fh:
        print("open file as file handler")
        while True:
            try:
                print("start each UTR region")
                start_t = timer()
                args = kwargs.copy()
                gene_info_str, df = pickle.load(fh)
                args["data"] = df
                args["gene_info_str"] = gene_info_str
                res = subsample_run(**args)
                end_t = timer()
                print(f"Done {gene_info_str} in {(end_t - start_t)/60} min.")
                res_lst.append(res)
            except EOFError:
                break

    with open(pickle_output_file, 'wb') as fh:
        for res in res_lst:
            print(f"save result of {res.gene_info_str}")
            pickle.dump(res, fh)


def test_run():
    run_mode = True
    res_file = 'tmp.pkl'
    if run_mode:
        start_t1 = timer()
        tbl = pd.read_table("../data/ENSMUSG00000000827_V7.txt",
                            names=['x', 'l', 'r', 'v4', 'v5', 's', 'pa'])

        tbl = tbl[["x", "l", "s", "pa"]]
        tbl.columns = ["x", "l", "r", "pa"]
        inds = np.sort(np.random.choice(len(tbl), 1000))
        utr_len = max(tbl["x"]) + max(tbl["l"]) + 50

        res, apamodel = subsample_run(output_file='tmpres.pkl', return_model=True, max_sample=200000, n_max_apa=4,
                                      data=tbl,
                                      utr_length=utr_len, debug=False)
        res.avg_pa_len = exp_pa_len(res, res.label_arr)
        print(f'avg_pa_len={res.avg_pa_len}')
        # apamodel, res = subsample_run(output_file='tmpres.pkl', max_sample=2000, n_max_apa=4, data=tbl.loc[inds], utr_length=utr_len, debug=False)

        # apamodel = ApaModel(n_max_apa=4, data=tbl.loc[inds], utr_length=utr_len, debug=False)
        # res = apamodel.run()
        # with open(res_file, 'wb') as fh:
        #    pickle.dump([apamodel, res], fh)
        end_t1 = timer()
        print(f'Full running time = {(end_t1 - start_t1) / 60} minutes.')
        print(f'bic = {res.bic}')
        tmpuniq, tmpcnt = np.unique(res.label_arr, return_counts=True)
        print("ws estimated from label_arr")
        print(tmpcnt / sum(tmpcnt))
        print("difference with ws")
        print(tmpcnt / sum(tmpcnt) - res.ws)

        print(f"orig_len={len(tbl)} bin_len={len(apamodel.x_arr)} res_label_len={len(res.label_arr)}")

        cov_prof = apamodel.coverage_profile
        plot_para(res)
        plt.plot(cov_prof[0], cov_prof[1] / np.sum(cov_prof[1]), label='coverage')
        # plt.show()
        plt.savefig("gpu_res.jpg")
    else:
        with open(res_file, 'rb') as fh:
            apamodel, res = pickle.load(fh)


# if __name__=="__main__":
#     infer_pa("../../test.pkl", ".")


# if __name__=="__main__":
#     exit_event = Event()
#     log_file = "./log.txt"
#     test_run_with_watchdog = watch_dog(log_file, exit_event)(test_run)
#     test_run_with_watchdog()







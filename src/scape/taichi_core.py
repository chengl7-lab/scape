# from math import log, exp, sqrt
import numpy as np
from itertools import product
import taichi as ti
import taichi.math as tm

pos_infinite = np.finfo('f').max  # 3.4028235e+38
neg_infinite = np.finfo('f').min  # -3.4028235e+38
PI = 3.141592653589793

ti.init(arch=ti.cuda, default_fp=ti.f64)
if ti.cfg.arch == ti.cuda:
    print("GPU is available")
else:
    print("GPU is not available")

"""
Author: Guangzhao Cheng, Lu Cheng
Date: 22.06.2023
"""


# ------------- base functions -------------
@ti.func
def my_log_taichi(x):
    log_x = tm.log(x)
    if x <= 0.0:
        log_x = neg_infinite
    return log_x

@ti.func
def logpdf_normal_taichi(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - tm.log(sigma) - 0.5 * tm.log(2 * PI)

@ti.func
def pdf_normal_taichi(x, mu, sigma):
    return tm.exp(-0.5 * ((x - mu) / sigma) ** 2) / tm.sqrt(2 * PI) / sigma

# Attention: different with logsumexp, the taichi func input a 2D array
@ti.func
def logsumexp_taichi(x_mat: ti.types.ndarray(), idx: int):
    n = x_mat.shape[1]
    max = x_mat[idx, 0]
    sum = 0.0

    ti.loop_config(serialize=True)
    for i in range(n):
        if x_mat[idx, i] > max:
            max = x_mat[idx, i]

    ti.loop_config(serialize=True)
    for i in range(n):
        sum += tm.exp(x_mat[idx, i] - max)
    return tm.log(sum) + max

@ti.func
def loglik_l_xt_taichi(x, l, theta):
    utr_len = theta - x
    res = neg_infinite
    if l <= utr_len:
        res = -tm.log(utr_len)
    return res

@ti.func
def lik_l_xt_taichi(x, l, theta) -> ti.f64:
    utr_len = theta - x
    res = 0.0
    if l <= utr_len:
        res = 1 / utr_len
    return res

@ti.func
def loglik_x_st_pa_taichi(pa, theta, sigma_f):
    return logpdf_normal_taichi(pa - theta, 0, sigma_f)


@ti.func
def loglik_x_st_taichi(x, s, theta, mu_f, sigma_f):
    return logpdf_normal_taichi(x, theta + s - mu_f, sigma_f)

@ti.func
def lik_x_st_taichi(x, s, theta, mu_f, sigma_f):
    return pdf_normal_taichi(x, theta + s - mu_f, sigma_f)

@ti.func
def loglik_r_s_taichi(r, s):
    res = neg_infinite
    if r <= s:
        res = -tm.log(s)
    return res

@ti.func
def lik_r_s_taichi(r, s):
    res = 0.0
    if r <= s:
        res = 1 / s
    return res

# ------------- kernel functions -------------
# ------------- [core func 1] loglik_xlr_t_pa -------------
@ti.kernel
def loglik_xlr_t_pa_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(),
                           pa_arr: ti.types.ndarray(), loglik_arr: ti.types.ndarray(),
                           theta: float, sigma_f: float, n_frag: int):
    for i in range(n_frag):
        loglik_arr[i] = loglik_l_xt_taichi(x_arr[i], l_arr[i], theta) + \
                        loglik_x_st_pa_taichi(pa_arr[i], theta, sigma_f)


# ------------- [core func 2] loglik_xlr_t_r_known -------------
@ti.kernel
def loglik_xlr_t_r_known_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(), r_arr: ti.types.ndarray(),
                                s_dis_arr: ti.types.ndarray(), pmf_s_arr: ti.types.ndarray(),
                                logpmf_s_arr: ti.types.ndarray(),
                                loglik_arr: ti.types.ndarray(),
                                tmp_mat: ti.types.ndarray(),
                                theta: float, mu_f: float, sigma_f: float):
    n_s = s_dis_arr.shape[0]
    n_frag = loglik_arr.shape[0]

    for i in range(n_frag):
        tmpn = 0.0
        for j in range(n_s):
            s = s_dis_arr[j]
            if s < r_arr[i]:
                tmp_mat[i, j] = neg_infinite
                continue
            else:
                tmpn += pmf_s_arr[j]
                tmp_mat[i, j] = loglik_r_s_taichi(r_arr[i], s) + loglik_x_st_taichi(x_arr[i], s, theta, mu_f, sigma_f) \
                                + loglik_l_xt_taichi(x_arr[i], l_arr[i], theta) + logpmf_s_arr[j]
        loglik_arr[i] = logsumexp_taichi(tmp_mat, i) - tm.log(tmpn)


# ------------- [core func 3] loglik_xlr_t_r_unknown -------------
# befor my_log
# cpu [0. ]            taichi [2.16908322e-309]
# after my_log
# cpu [neg_infinite]   taichi [-710.7244891]

@ti.kernel
def loglik_xlr_t_r_unknown_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(), r_arr: ti.types.ndarray(),
                                  s_dis_arr: ti.types.ndarray(), pmf_s_arr: ti.types.ndarray(),
                                  loglik_arr: ti.types.ndarray(),
                                  theta: float, mu_f: float, sigma_f: float, n_frag: int, n_s: int):
    for i in range(n_frag):
        for j in range(n_s):
            s = s_dis_arr[j]
            loglik_arr[i] += 1 / s * lik_x_st_taichi(x_arr[i], s, theta, mu_f, sigma_f) * lik_l_xt_taichi(x_arr[i],
                                                                                                          l_arr[i],
                                                                                                          theta) * \
                             pmf_s_arr[j]
        # ----------- add 2023.04.20 ----------------
        if loglik_arr[i] < 1e-300:
            loglik_arr[i] = 0.0
        # ----------- add end 2023.04.20 ------------
        loglik_arr[i] = my_log_taichi(loglik_arr[i])

# ------------- [core func 4]  loglik_marginal_lxr -------------
@ti.kernel
def call_logp_theta_sum_kernel(all_theta: ti.types.ndarray(), logp_theta_arr: ti.types.ndarray(),
                               n_sel_theta: int, alpha: float, beta: float, min_ind: int) -> ti.f64:
    p_theta_sum = 0.0
    for i in range(n_sel_theta):
        it = i + min_ind
        logp_theta_arr[i] = logpdf_normal_taichi(all_theta[it], alpha, beta)
        p_theta_sum += tm.exp(logp_theta_arr[i])
    logp_theta_sum = tm.log(p_theta_sum)
    return logp_theta_sum


@ti.kernel
def cal_res_kernel(loglik_xlr_t_arr: ti.types.ndarray(), logp_theta_arr: ti.types.ndarray(), res: ti.types.ndarray(),
                   tmp_mat: ti.types.ndarray(), logp_theta_sum: float, n_sel_theta: int, min_ind: int, n_frag: int):
    for j in range(n_frag):
        for i in range(n_sel_theta):
            it = i + min_ind
            tmp_mat[j, i] = loglik_xlr_t_arr[j, it] + logp_theta_arr[i] - logp_theta_sum
        res[j] = logsumexp_taichi(tmp_mat, j)


# ------------- interface functions -------------
def loglik_xlr_t_pa(x_arr, l_arr, pa_arr, theta, sigma_f):
    """
    Args:
        x_arr:  NumPy array, float64, (n_frag,)
        l_arr: NumPy array, float64, (n_frag,)
        pa_arr: NumPy array, float64, (n_frag,)
        theta: float64
        sigma_f: int
    Returns:
        loglik_arr: NumPy array, float64, (n_frag,)
    """
    n_frag = len(x_arr)
    loglik_arr = np.zeros(n_frag)
    loglik_xlr_t_pa_kernel(x_arr, l_arr, pa_arr, loglik_arr, theta, sigma_f, n_frag)
    return loglik_arr


def loglik_xlr_t_r_known(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    loglik_arr = np.zeros(n_frag)
    logpmf_s_arr = np.log(pmf_s_arr)
    tmp_mat = np.zeros((n_frag, n_s)) + neg_infinite
    loglik_xlr_t_r_known_kernel(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, logpmf_s_arr, loglik_arr,
                                tmp_mat, theta, mu_f, sigma_f)
    return loglik_arr


def loglik_xlr_t_r_unknown(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    loglik_arr = np.zeros(n_frag)
    loglik_xlr_t_r_unknown_kernel(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, loglik_arr,
                                  theta, mu_f, sigma_f, n_frag, n_s)
    return loglik_arr


def loglik_marginal_lxr(alpha, beta, all_theta, loglik_xlr_t_arr):
    n_frag = loglik_xlr_t_arr.shape[0]

    min_ind = int(np.searchsorted(all_theta, alpha - 3 * beta, side='left'))
    max_ind = int(np.searchsorted(all_theta, alpha + 3 * beta, side='right') - 1)

    n_sel_theta = max_ind - min_ind + 1
    res = np.zeros(n_frag)
    logp_theta_arr = np.zeros(n_sel_theta)

    # ti@kernel 1
    logp_theta_sum = call_logp_theta_sum_kernel(all_theta, logp_theta_arr,
                               n_sel_theta, alpha, beta, min_ind)

    tmp_mat = np.zeros((n_frag, n_sel_theta)) + neg_infinite
    cal_res_kernel(loglik_xlr_t_arr, logp_theta_arr, res, tmp_mat, logp_theta_sum, n_sel_theta, min_ind, n_frag)
    return res  # n_frag x 1


def get_loglik_marginal_tensor(all_theta, predef_beta_arr, loglik_xlr_t_arr):
    n_alpha = len(all_theta)
    n_beta = len(predef_beta_arr)
    n_frag = loglik_xlr_t_arr.shape[0]
    res = np.zeros((n_alpha, n_beta, n_frag)) + neg_infinite

    for i, j in product(range(n_alpha), range(n_beta)):
        res[i, j] = loglik_marginal_lxr(all_theta[i], predef_beta_arr[j], all_theta, loglik_xlr_t_arr)

    return res

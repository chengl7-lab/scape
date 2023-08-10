from math import log, exp, sqrt
import numpy as np
from itertools import product
import taichi as ti
# import taichi.math as tm

from taichi_core import my_log_taichi, logpdf_normal_taichi, pdf_normal_taichi, logsumexp_taichi, loglik_l_xt_taichi, \
    lik_l_xt_taichi, loglik_x_st_pa_taichi, loglik_x_st_taichi, lik_x_st_taichi, loglik_r_s_taichi, lik_r_s_taichi, \
    loglik_xlr_t_r_known_kernel, loglik_xlr_t_r_unknown_kernel, loglik_xlr_t_pa, call_logp_theta_sum_kernel, cal_res_kernel

"""
Author: Guangzhao Cheng, Lu Cheng
Date: 22.06.2023
"""

pos_infinite = np.finfo('f').max  # 3.4028235e+38
neg_infinite = np.finfo('f').min  # -3.4028235e+38
PI = 3.141592653589793

ti.init(arch=ti.cuda, default_fp=ti.f64)
# ti.init(arch=ti.cpu, default_fp=ti.f64)


# ------------- base functions -------------

def my_log(x):
    if x <= 0.0:
        return neg_infinite
    else:
        return log(x)


# @ti.func
# def my_log_taichi(x):
#     log_x = tm.log(x)
#     if x <= 0.0:
#         log_x = neg_infinite
#     return log_x


@ti.kernel
def my_log_taichi_test_kernel(n: float) -> float:
    return my_log_taichi(n)


def logpdf_normal(x, mu, sigma):
    """
    Compute the log PDF of a normal distribution with mean `mu` and standard deviation `sigma`
    at a given value `x`.
    """
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma) - 0.5 * log(2 * PI)


# @ti.func
# def logpdf_normal_taichi(x, mu, sigma):
#     return -0.5 * ((x - mu) / sigma) ** 2 - tm.log(sigma) - 0.5 * tm.log(2 * PI)


@ti.kernel
def logpdf_normal_taichi_test_kernel(x: float, mu: float, sigma: float) -> float:
    return logpdf_normal_taichi(x, mu, sigma)


def pdf_normal(x, mu, sigma):
    """
    Compute the log PDF of a normal distribution with mean `mu` and standard deviation `sigma`
    at a given value `x`.
    """
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / sqrt(2 * PI) / sigma


# @ti.func
# def pdf_normal_taichi(x, mu, sigma):
#     return tm.exp(-0.5 * ((x - mu) / sigma) ** 2) / tm.sqrt(2 * PI) / sigma


@ti.kernel
def pdf_normal_taichi_test_kernel(x: float, mu: float, sigma: float) -> ti.f64:
    return pdf_normal_taichi(x, mu, sigma)


def logsumexp(x_arr):
    n = len(x_arr)
    max = x_arr[0]
    sum = 0.0

    for i in range(n):
        if x_arr[i] > max:
            max = x_arr[i]

    for i in range(n):
        sum = sum + exp(x_arr[i] - max)
    return log(sum) + max


# # Attention: different with logsumexp, the taichi func input a 2D array
# @ti.func
# def logsumexp_taichi(x_mat: ti.types.ndarray(), idx: int):
#     n = x_mat.shape[1]
#     max = x_mat[idx, 0]
#     sum = 0.0
#
#     ti.loop_config(serialize=True)
#     for i in range(n):
#         if x_mat[idx, i] > max:
#             max = x_mat[idx, i]
#
#     ti.loop_config(serialize=True)
#     for i in range(n):
#         sum += tm.exp(x_mat[idx, i] - max)
#     return tm.log(sum) + max


@ti.kernel
def logsumexp_taichi_test_kernel(x_arr: ti.types.ndarray(), idx: int) -> ti.f64:
    return logsumexp_taichi(x_arr, idx)


def loglik_l_xt(x, l, theta):
    utr_len = theta - x

    if l <= utr_len:
        return -log(utr_len)
    else:
        return neg_infinite


# @ti.func
# def loglik_l_xt_taichi(x, l, theta):
#     utr_len = theta - x
#     res = neg_infinite
#     if l <= utr_len:
#         res = -tm.log(utr_len)
#     return res


@ti.kernel
def loglik_l_xt_taichi_test_kernel(x: float, l: float, theta: float) -> float:
    return loglik_l_xt_taichi(x, l, theta)


def lik_l_xt(x, l, theta) -> float:
    utr_len = theta - x
    if l <= utr_len:
        return 1 / utr_len
    else:
        return 0.0

#
# @ti.func
# def lik_l_xt_taichi(x, l, theta) -> ti.f64:
#     utr_len = theta - x
#     res = 0.0
#     if l <= utr_len:
#         res = 1 / utr_len
#     return res


@ti.kernel
def lik_l_xt_taichi_test_kernel(x: float, l: float, theta: float) -> ti.f64:
    return lik_l_xt_taichi(x, l, theta)


def loglik_x_st_pa(pa, theta, sigma_f):
    return logpdf_normal(pa - theta, 0, sigma_f)

#
# @ti.func
# def loglik_x_st_pa_taichi(pa, theta, sigma_f):
#     return logpdf_normal_taichi(pa - theta, 0, sigma_f)


@ti.kernel
def loglik_x_st_pa_taichi_test_kernel(pa: float, theta: float, sigma_f: float) -> float:
    return loglik_x_st_pa_taichi(pa, theta, sigma_f)


def loglik_x_st(x, s, theta, mu_f, sigma_f):
    return logpdf_normal(x, theta + s - mu_f, sigma_f)


# @ti.func
# def loglik_x_st_taichi(x, s, theta, mu_f, sigma_f):
#     return logpdf_normal_taichi(x, theta + s - mu_f, sigma_f)


@ti.kernel
def loglik_x_st_taichi_test_kernel(x: float, s: float, theta: float, mu_f: float, sigma_f: float) -> float:
    return loglik_x_st_taichi(x, s, theta, mu_f, sigma_f)


def lik_x_st(x, s, theta, mu_f, sigma_f):
    return pdf_normal(x, theta + s - mu_f, sigma_f)


# @ti.func
# def lik_x_st_taichi(x, s, theta, mu_f, sigma_f):
#     return pdf_normal_taichi(x, theta + s - mu_f, sigma_f)


@ti.kernel
def lik_x_st_taichi_test_kernel(x: float, s: float, theta: float, mu_f: float, sigma_f: float) -> float:
    return lik_x_st_taichi(x, s, theta, mu_f, sigma_f)


def loglik_r_s(r, s):
    if r <= s:
        return -log(s)
    else:
        return neg_infinite


# @ti.func
# def loglik_r_s_taichi(r, s):
#     res = neg_infinite
#     if r <= s:
#         res = -tm.log(s)
#     return res


@ti.kernel
def loglik_r_s_taichi_test_kernel(r: float, s: float) -> float:
    return loglik_r_s_taichi(r, s)


def lik_r_s(r, s):
    if r <= s:
        return 1 / s
    else:
        return 0.0


# @ti.func
# def lik_r_s_taichi(r, s):
#     res = 0.0
#     if r <= s:
#         res = 1 / s
#     return res


@ti.kernel
def lik_r_s_taichi_test_kernel(r: float, s: float) -> float:
    return lik_r_s_taichi(r, s)


# ------------- [core func 1] loglik_xlr_t_pa -------------


def loglik_xlr_t_pa_cpu(x_arr, l_arr, pa_arr, theta, sigma_f):
    n_frag = len(x_arr)
    loglik_arr = np.zeros(n_frag)
    for i in range(n_frag):
        loglik_arr[i] = loglik_l_xt(x_arr[i], l_arr[i], theta) + loglik_x_st_pa(pa_arr[i], theta, sigma_f)
    return loglik_arr


@ti.kernel
def loglik_xlr_t_pa_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(),
                           pa_arr: ti.types.ndarray(), loglik_arr: ti.types.ndarray(),
                           theta: float, sigma_f: float, n_frag: int):
    for i in range(n_frag):
        loglik_arr[i] = loglik_l_xt_taichi(x_arr[i], l_arr[i], theta) + \
                        loglik_x_st_pa_taichi(pa_arr[i], theta, sigma_f)


# def loglik_xlr_t_pa(x_arr, l_arr, pa_arr, theta, sigma_f):
#     """
#     Args:
#         x_arr:  NumPy array, float64, (n_frag,)
#         l_arr: NumPy array, float64, (n_frag,)
#         pa_arr: NumPy array, float64, (n_frag,)
#         theta: float64
#         sigma_f: int
#     Returns:
#         loglik_arr: NumPy array, float64, (n_frag,)
#     """
#     n_frag = len(x_arr)
#     loglik_arr = np.zeros(n_frag)
#     loglik_xlr_t_pa_kernel(x_arr, l_arr, pa_arr, loglik_arr, theta, sigma_f, n_frag)
#     return loglik_arr
#
#
# # ------------- [core func 2] loglik_xlr_t_r_known -------------
#
def loglik_xlr_t_r_known_cpu(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    logpmf_s_arr = np.log(pmf_s_arr)
    loglik_arr = np.zeros(n_frag)
    tmp_arr = np.zeros(n_s) + neg_infinite
    for i in range(n_frag):
        tmpn = 0.0
        # tmpv = 0.0
        for j in range(n_s):
            s = s_dis_arr[j]
            if s < r_arr[i]:
                tmp_arr[j] = neg_infinite
                continue
            else:
                tmpn += pmf_s_arr[j]
                # tmpv += lik_r_s(r_arr[i], s) * lik_x_st(x_arr[i], s, theta, mu_f, sigma_f) *
                # lik_l_xt(x_arr[i], l_arr[i], theta) * pmf_s_arr[j]
                tmp_arr[j] = loglik_r_s(r_arr[i], s) + loglik_x_st(x_arr[i], s, theta, mu_f, sigma_f) + loglik_l_xt(
                    x_arr[i], l_arr[i], theta) + logpmf_s_arr[j]
        # loglik_arr[i] = my_log(tmpv/tmpn)
        loglik_arr[i] = logsumexp(tmp_arr) - log(tmpn)
        # print(i, "tmp_arr cpu : ", tmp_arr)
        # print(i, "loglik_arr cpu : ", loglik_arr[i])

    return loglik_arr


# @ti.kernel
# def loglik_xlr_t_r_known_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(), r_arr: ti.types.ndarray(),
#                                 s_dis_arr: ti.types.ndarray(), pmf_s_arr: ti.types.ndarray(),
#                                 logpmf_s_arr: ti.types.ndarray(),
#                                 loglik_arr: ti.types.ndarray(),
#                                 tmp_mat: ti.types.ndarray(),
#                                 theta: float, mu_f: float, sigma_f: float):
#     n_s = s_dis_arr.shape[0]
#     n_frag = loglik_arr.shape[0]
#
#     for i in range(n_frag):
#         tmpn = 0.0
#         for j in range(n_s):
#             s = s_dis_arr[j]
#             if s < r_arr[i]:
#                 tmp_mat[i, j] = neg_infinite
#                 continue
#             else:
#                 tmpn += pmf_s_arr[j]
#                 tmp_mat[i, j] = loglik_r_s_taichi(r_arr[i], s) + loglik_x_st_taichi(x_arr[i], s, theta, mu_f, sigma_f) \
#                                 + loglik_l_xt_taichi(x_arr[i], l_arr[i], theta) + logpmf_s_arr[j]
#         loglik_arr[i] = logsumexp_taichi(tmp_mat, i) - tm.log(tmpn)
#
#
def loglik_xlr_t_r_known(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    loglik_arr = np.zeros(n_frag)
    logpmf_s_arr = np.log(pmf_s_arr)
    tmp_mat = np.zeros((n_frag, n_s)) + neg_infinite
    loglik_xlr_t_r_known_kernel(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, logpmf_s_arr, loglik_arr,
                                tmp_mat, theta, mu_f, sigma_f)
    return loglik_arr


# # ------------- [core func 3] loglik_xlr_t_r_unknown -------------
# # befor my_log
# # cpu [0. ]            taichi [2.16908322e-309]
# # after my_log
# # cpu [neg_infinite]   taichi [-710.7244891]
#
#
def loglik_xlr_t_r_unknown_cpu(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    loglik_arr = np.zeros(n_frag)
    for i in range(n_frag):
        for j in range(n_s):
            s = s_dis_arr[j]
            loglik_arr[i] += 1/s * lik_x_st(x_arr[i], s, theta, mu_f, sigma_f) * lik_l_xt(x_arr[i], l_arr[i], theta) * pmf_s_arr[j]
        # ----------- add 2023.04.20 ----------------
        if loglik_arr[i] < 1e-300:
            loglik_arr[i] = 0.0
        # ----------- add end 2023.04.20 ------------
        loglik_arr[i] = my_log(loglik_arr[i])
    return loglik_arr


# @ti.kernel
# def loglik_xlr_t_r_unknown_kernel(x_arr: ti.types.ndarray(), l_arr: ti.types.ndarray(), r_arr: ti.types.ndarray(),
#                                   s_dis_arr: ti.types.ndarray(), pmf_s_arr: ti.types.ndarray(),
#                                   loglik_arr: ti.types.ndarray(),
#                                   theta: float, mu_f: float, sigma_f: float, n_frag: int, n_s: int):
#     for i in range(n_frag):
#         for j in range(n_s):
#             s = s_dis_arr[j]
#             loglik_arr[i] += 1 / s * lik_x_st_taichi(x_arr[i], s, theta, mu_f, sigma_f) * lik_l_xt_taichi(x_arr[i],
#                                                                                                           l_arr[i],
#                                                                                                           theta) * \
#                              pmf_s_arr[j]
#         # ----------- add 2023.04.20 ----------------
#         if loglik_arr[i] < 1e-300:
#             loglik_arr[i] = 0.0
#         # ----------- add end 2023.04.20 ------------
#         loglik_arr[i] = my_log_taichi(loglik_arr[i])
#
#
def loglik_xlr_t_r_unknown(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f):
    n_frag, n_s = len(x_arr), len(s_dis_arr)
    loglik_arr = np.zeros(n_frag)
    loglik_xlr_t_r_unknown_kernel(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, loglik_arr,
                                  theta, mu_f, sigma_f, n_frag, n_s)
    return loglik_arr


# ------------- [core func 4]  loglik_marginal_lxr -------------


def loglik_marginal_lxr_cpu(alpha, beta, all_theta, loglik_xlr_t_arr):
    n_theta = len(all_theta)
    n_frag = loglik_xlr_t_arr.shape[0]

    min_ind = pos_infinite
    max_ind = neg_infinite
    for i in range(n_theta):
        if all_theta[i] >= alpha - 3 * beta and all_theta[i] <= alpha + 3 * beta:
            if i < min_ind:
                min_ind = i
            if i > max_ind:
                max_ind = i

    n_sel_theta = max_ind - min_ind + 1
    res = np.zeros(n_frag)
    logp_theta_arr = np.zeros(n_sel_theta)

    p_theta_sum = 0.0
    for i in range(n_sel_theta):
        it = i + min_ind
        logp_theta_arr[i] = logpdf_normal(all_theta[it], alpha, beta)
        p_theta_sum += exp(logp_theta_arr[i])
    logp_theta_sum = log(p_theta_sum)

    tmp_arr = np.zeros(n_sel_theta) + neg_infinite
    for j in range(n_frag):
        for i in range(n_sel_theta):
            it = i + min_ind
            # res[j] += exp(loglik_xlr_t_arr[j, it] + logp_theta_arr[i])/p_theta_sum
            tmp_arr[i] = loglik_xlr_t_arr[j, it] + logp_theta_arr[i] - logp_theta_sum
        # res[j] = my_log(res[j])
        res[j] = logsumexp(tmp_arr)
    return res  # n_frag x 1


# @ti.kernel
# def call_logp_theta_sum_kernel(all_theta: ti.types.ndarray(), logp_theta_arr: ti.types.ndarray(),
#                                n_sel_theta: int, alpha: float, beta: float, min_ind: int) -> ti.f64:
#     p_theta_sum = 0.0
#     for i in range(n_sel_theta):
#         it = i + min_ind
#         logp_theta_arr[i] = logpdf_normal_taichi(all_theta[it], alpha, beta)
#         p_theta_sum += tm.exp(logp_theta_arr[i])
#     logp_theta_sum = tm.log(p_theta_sum)
#     return logp_theta_sum
#
#
# @ti.kernel
# def cal_res_kernel(loglik_xlr_t_arr: ti.types.ndarray(), logp_theta_arr: ti.types.ndarray(), res: ti.types.ndarray(),
#                    tmp_mat: ti.types.ndarray(), logp_theta_sum: float, n_sel_theta: int, min_ind: int, n_frag: int):
#     for j in range(n_frag):
#         for i in range(n_sel_theta):
#             it = i + min_ind
#             tmp_mat[j, i] = loglik_xlr_t_arr[j, it] + logp_theta_arr[i] - logp_theta_sum
#         res[j] = logsumexp_taichi(tmp_mat, j)
#
#
def loglik_marginal_lxr(alpha, beta, all_theta, loglik_xlr_t_arr):
    n_frag = loglik_xlr_t_arr.shape[0]

    # n_theta = len(all_theta)
    # min_ind = pos_infinite
    # max_ind = neg_infinite
    # for i in range(n_theta):
    #     if all_theta[i] >= alpha - 3 * beta and all_theta[i] <= alpha + 3 * beta:
    #         if i < min_ind:
    #             min_ind = i
    #         if i > max_ind:
    #             max_ind = i
    #     elif all_theta[i] > alpha + 3 * beta:
    #         break

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


# ------------- test functions -------------


def test_all_base_func(test_threshold=1e-40):

    print("\n" + 10 * "-", " test base functions ( test_threshold =", test_threshold, ")", 10 * "-")

    def print_test(_res_cpu, _res_gpu, info):
        if abs(_res_cpu - _res_gpu) < test_threshold:
            print(info, " success !")
        else:
            print(info, " error !!!")
            print("cpu    : ", _res_cpu)
            print("taichi : ", _res_gpu)
            print()

    # ------------- my_log v.s. my_log_taichi -------------
    x = 3.4567
    res_cpu = my_log(x)
    res_gpu = my_log_taichi_test_kernel(x)
    print_test(res_cpu, res_gpu, "* my_log test 1 :")

    x = - 3.4567
    res_cpu = my_log(x)
    res_gpu = my_log_taichi_test_kernel(x)
    print_test(res_cpu, res_gpu, "* my_log test 2 :")

    # ------------- logpdf_normal v.s. logpdf_normal_taichi -------------
    res_cpu = logpdf_normal(0.75, 1.0, 0.5)
    res_gpu = logpdf_normal_taichi_test_kernel(0.75, 1.0, 0.5)
    print_test(res_cpu, res_gpu, "* logpdf_normal :")

    # ------------- pdf_normal v.s. pdf_normal_taichi -------------
    res_cpu = pdf_normal(0.75, 1.0, 0.5)
    res_gpu = pdf_normal_taichi_test_kernel(0.75, 1.0, 0.5)
    print_test(res_cpu, res_gpu, "* pdf_normal :")

    # ------------- logsumexp v.s. logsumexp_taichi -------------
    x_arr = np.array([3.65, 7.89, 5., 6.12, 1.23])  # float64
    res_cpu = logsumexp(x_arr)
    x_arr = np.array([[3.65, 7.89, 5., 6.12, 1.23], [0., 0., 0., 0., 0.]])  # float64
    res_gpu = logsumexp_taichi_test_kernel(x_arr, 0)
    print_test(res_cpu, res_gpu, "* logsumexp test 1:")
    # res_gpu = logsumexp_taichi_test_kernel(x_arr, 1)
    # print_test(res_cpu, res_gpu, "* logsumexp test 2:")

    # ------------- loglik_l_xt v.s. loglik_l_xt_taichi -------------
    res_cpu = loglik_l_xt(30, 50, 70)
    res_gpu = loglik_l_xt_taichi_test_kernel(30, 50, 70)
    print_test(res_cpu, res_gpu, "* loglik_l_xt test 1 :")

    res_cpu = loglik_l_xt(5, 50, 70)
    res_gpu = loglik_l_xt_taichi_test_kernel(5, 50, 70)
    print_test(res_cpu, res_gpu, "* loglik_l_xt test 2 :")

    # ------------- lik_l_xt v.s. lik_l_xt_taichi -------------
    res_cpu = lik_l_xt(30, 50, 70)
    res_gpu = lik_l_xt_taichi_test_kernel(30, 50, 70)
    print_test(res_cpu, res_gpu, "* lik_l_xt test 1 :")
    #
    res_cpu = lik_l_xt(5, 50, 70)
    res_gpu = lik_l_xt_taichi_test_kernel(5, 50, 70)
    print_test(res_cpu, res_gpu, "* lik_l_xt test 2 :")

    # ------------- loglik_x_st_pa v.s. loglik_x_st_pa_taichi -------------
    res_cpu = loglik_x_st_pa(187, 460, 50)
    res_gpu = loglik_x_st_pa_taichi_test_kernel(187, 460, 50)
    print_test(res_cpu, res_gpu, "* loglik_x_st_pa :")

    # ------------- loglik_x_st v.s. loglik_x_st_taichi -------------
    res_cpu = loglik_x_st(44, 144, 460, 50, 50)
    res_gpu = loglik_x_st_taichi_test_kernel(44, 144, 460, 50, 50)
    print_test(res_cpu, res_gpu, "* loglik_x_st :")

    # ------------- lik_x_st_pa v.s. lik_x_st_pa_taichi -------------
    res_cpu = lik_x_st(44, 144, 460, 50, 50)
    res_gpu = lik_x_st_taichi_test_kernel(44, 144, 460, 50, 50)
    print_test(res_cpu, res_gpu, "* lik_x_st_pa :")

    # ------------- loglik_r_s v.s. loglik_r_s_taichi -------------
    res_cpu = loglik_r_s(10, 20)
    res_gpu = loglik_r_s_taichi_test_kernel(10, 20)
    print_test(res_cpu, res_gpu, "* loglik_r_s test 1 :")

    res_cpu = loglik_r_s(20, 10)
    res_gpu = loglik_r_s_taichi_test_kernel(20, 10)
    print_test(res_cpu, res_gpu, "* loglik_r_s test 2 :")

    # ------------- lik_r_s v.s. lik_r_s_taichi -------------
    res_cpu = lik_r_s(10, 20)
    res_gpu = lik_r_s_taichi_test_kernel(10, 20)
    print_test(res_cpu, res_gpu, "* lik_r_s test 1 :")

    res_cpu = lik_r_s(20, 10)
    res_gpu = lik_r_s_taichi_test_kernel(20, 10)
    print_test(res_cpu, res_gpu, "* lik_r_s test 2 :")


def test_loglik_xlr_t_pa(test_data_file_name, test_threshold=1e-40):

    print("\n" + 10 * "-", " test_loglik_xlr_t_pa ( test_threshold =", test_threshold, ")",  10 * "-")

    # read test data
    with open(test_data_file_name, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(",")
            if i == 0:
                x_arr = np.array([float(x) for x in line])
            elif i == 1:
                l_arr = np.array([float(x) for x in line])
            elif i == 2:
                pa_arr = np.array([float(x) for x in line])
            elif i == 3:
                theta = float(line[0])
            elif i == 4:
                sigma_f = int(line[0])
            # elif i == 5:
            #     loglik_arr = np.array([float(x) for x in line])
            else:
                break
    theta_arr = [theta, theta-5, theta+5]

    for i in range(3):


        # cpu res
        estimate_cpu_loglik_arr = loglik_xlr_t_pa_cpu(x_arr, l_arr, pa_arr, theta_arr[i], sigma_f)

        # taichi res
        estimate_taichi_loglik_arr = loglik_xlr_t_pa(x_arr, l_arr, pa_arr, theta_arr[i], sigma_f)

        res = np.sum(np.abs(estimate_cpu_loglik_arr - estimate_taichi_loglik_arr))

        if res < test_threshold:
            print("* test case ", i, " success !, error_sum = ", res)
        else:
            print("* test case ", i, " error !!!  error_sum = ", res)


def test_loglik_xlr_t_r_known(test_data_file_name, test_threshold=1e-40):

    print("\n" + 10 * "-", " test_loglik_xlr_t_r_known ( test_threshold =", test_threshold, ")", 10 * "-")

    # read test data
    # (x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f)
    with open(test_data_file_name, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(",")
            if i == 0:
                x_arr = np.array([float(x) for x in line])
            elif i == 1:
                l_arr = np.array([float(x) for x in line])
            elif i == 2:
                r_arr = np.array([float(x) for x in line])
            elif i == 3:
                s_dis_arr = np.array([float(x) for x in line[0:13]])
            elif i == 4:
                pmf_s_arr = np.array([float(x) for x in line[0:13]])
            elif i == 5:
                theta = float(line[0])
            elif i == 6:
                mu_f = float(line[0])
            elif i == 7:
                sigma_f = float(line[0])
            else:
                break
    theta_arr = [theta, theta-5, theta+5]

    for i in range(3):
        # cpu res
        estimate_cpu_loglik_arr = loglik_xlr_t_r_known_cpu(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta_arr[i],
                                                           mu_f, sigma_f)

        # taichi res
        estimate_taichi_loglik_arr = loglik_xlr_t_r_known(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta_arr[i],
                                                          mu_f, sigma_f)
        res = np.abs(estimate_cpu_loglik_arr - estimate_taichi_loglik_arr)
        res = np.sum(res)
        if res < test_threshold:
            print("* test case ", i, " success !, error_sum = ", res)
        else:
            print("* test case ", i, " error !!!  error_sum = ", res)


def test_loglik_xlr_t_r_unknown(test_data_file_name, test_threshold=1e-40):

    print("\n" + 10 * "-", " test_loglik_xlr_t_r_unknown ( test_threshold =", test_threshold, ")", 10 * "-")

    # read test data
    # (x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta, mu_f, sigma_f)
    with open(test_data_file_name, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(",")
            if i == 0:
                x_arr = np.array([float(x) for x in line])
            elif i == 1:
                l_arr = np.array([float(x) for x in line])
            elif i == 2:
                r_arr = np.array([])
            elif i == 3:
                s_dis_arr = np.array([float(x) for x in line[0:13]])
            elif i == 4:
                pmf_s_arr = np.array([float(x) for x in line[0:13]])
            elif i == 5:
                theta = float(line[0])
            elif i == 6:
                mu_f = float(line[0])
            elif i == 7:
                sigma_f = float(line[0])
            else:
                break
    theta_arr = [theta, theta-5, theta+5]

    for i in range(3):
        # cpu res
        estimate_cpu_loglik_arr = loglik_xlr_t_r_unknown_cpu(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta_arr[i],
                                                           mu_f, sigma_f)

        # taichi res
        estimate_taichi_loglik_arr = loglik_xlr_t_r_unknown(x_arr, l_arr, r_arr, s_dis_arr, pmf_s_arr, theta_arr[i],
                                                          mu_f, sigma_f)
        res = np.abs(estimate_cpu_loglik_arr - estimate_taichi_loglik_arr)

        res = np.sum(res)
        if res < test_threshold:
            print("* test case ", i, " success !, error_sum = ", res)
        else:
            print("* test case ", i, " error !!!  error_sum = ", res)

        res = np.abs(estimate_cpu_loglik_arr / estimate_taichi_loglik_arr - 1.0)

        max_error_r_unknown = np.nanmax(res)

        if max_error_r_unknown < 1e-3:
            print("* test case ", i, " success! (test_test_threshold=1e-3) max_error_r_unknown = ", max_error_r_unknown)
        else:
            print("* test case ", i, " error!!! (test_test_threshold=1e-3) max_error_r_unknown = ", max_error_r_unknown)


def test_loglik_marginal_lxr(test_threshold=1e-40):

    print("\n" + 10 * "-", " test_loglik_marginal_lxr ( test_threshold =", test_threshold, ")", 10 * "-")

    alpha = 37
    beta = 5
    all_theta = np.array([37, 46, 55, 64, 73, 82, 91, 100, 109, 118, 127, 136, 145, 154,
                          163, 172, 181, 190, 199, 208, 217, 226, 235])

    for i in range(3):
        loglik_xlr_t_arr = np.random.rand(i+3, i+5)

        cpu_res = loglik_marginal_lxr_cpu(alpha, beta, all_theta, loglik_xlr_t_arr)
        # print("cpu_res : ", cpu_res)

        gpu_res = loglik_marginal_lxr(alpha, beta, all_theta, loglik_xlr_t_arr)
        # print("gpu_res : ", gpu_res)

        res = np.abs(cpu_res - gpu_res)
        res = np.sum(res)
        if res < test_threshold:
            print("* test case ", i, " success !, error_sum = ", res)
        else:
            print("* test case ", i, " error !!!  error_sum = ", res)


if __name__ == "__main__":
    test_all_base_func()
    test_loglik_xlr_t_pa("../data/test_data1.csv")
    test_loglik_xlr_t_r_known("../data/test_data2.csv")
    test_loglik_xlr_t_r_unknown("../data/test_data3.csv")
    test_loglik_marginal_lxr()
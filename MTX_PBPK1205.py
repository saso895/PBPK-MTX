import numpy as np
from scipy.integrate import odeint
import pandas as pd
from pars1111 import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
from pars1111 import params
import warnings
import pymc as pm
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
#from scipy.interpolate import interp1d

init_pars = [params["PRest"], params["PK"], params["PL"], params["Kbile"], params["GFR"],
                 params["Free"], params["Vmax_baso"], params["Km_baso"], params["Kurine"],
             params["Kreab"]]

# 初始化计数器
infusion_usage_count = 0  # 记录infusion_rate参与计算的次数
time_points = []
derivs_call_count = 0
def derivs(y, time, pars, DOSEdata):
    #print(f"derivs_time : {time}")
    # Unpack parameters and state variables
    global infusion_usage_count
    global derivs_call_count
    global time_points
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine,Kreab = pars
    # Unpack state variables
    APlas=y[0]
    AL=y[1]
    AK=y[2]
    ARest=y[3]
    Atubu=y[4]
    Aurine=y[5]
    Afeces=y[6]
    time_points.append(time)
    # Extract infusion time and rate
    infusion_time=DOSEdata['time'].values[0]
    Rate=DOSEdata['Rate'].values[0]
    infusion_rate = Rate if time <= infusion_time else 0
    #print(f"infusion_time : {infusion_time},obe_time : {time},infusion_rate : {Rate}")
    # Define differential equations
    dAPlas = (QRest * ARest / VRest / PRest) + (QK * AK / VK / PK) + (QL * AL / VL / PL) - (
            QPlas * APlas / VPlas) + Kreab * Atubu + infusion_rate
    dAL = QL * (APlas / VPlas - AL / VL / PL) - Kbile * AL
    dAK = QK * (APlas / VPlas - AK / VK / PK) - (APlas / VPlas) * GFR * Free - (Vmax_baso * AK / VK / PK) / (
            Km_baso + AK / VK / PK)
    dARest = QRest * (APlas / VPlas - ARest / VRest / PRest)
    dAtubu = (APlas / VPlas) * GFR * Free + (Vmax_baso * AK / VK / PK) / (
            Km_baso + AK / VK / PK) - Atubu * Kurine - Kreab * Atubu
    dAurine = Kurine * Atubu
    dAfeces = Kbile * AL
    # Pack derivatives and other outputs
    dydt = [dAPlas, dAL, dAK, dARest, dAtubu, dAurine, dAfeces]
    # 打印额外的调试信息
    #print(f"Time: {time}, APlas: {APlas},infusion_rate:{infusion_rate}")
        #f", AL: {AL}, AK: {AK}, ARest: {ARest}, Atubu: {Atubu}, Aurine: {Aurine}, Afeces: {Afeces}")

    return dydt

def MTX_PBPK(pars, Time_point, DOSEdata):

    Time = Time_point
    #time_toal = Time_point[-1]
    global infusion_usage_count
    # 1.B) Specify initial conditions
    param=exp_denormalize(pars)
    #PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine,Kreab=  param

    # Set initial conditions according to mass balance
    APlas_0 = 0
    AL_0 = 0
    AK_0 = 0
    ARest_0 = 0
    Atubu_0 = 0
    Aurine_0 = 0
    Afeces_0 = 0

    # Specify initial conditions
    y0 = [APlas_0, AL_0, AK_0, ARest_0, Atubu_0, Aurine_0, Afeces_0]

    out = odeint(
        derivs,
        y0,
        Time,
        args=( param, DOSEdata),
        hmax=1,
        atol=1e-3,
        rtol=1e-4)
    print("所有时间点:", np.unique(time_points[-1]))
    # 设置阈值
    threshold = 1e-6  # 你可以根据需要调整这个阈值

    # 后处理：将小于阈值的值设置为0
    for i in range(len(out)):
        for j in range(len(out[i])):
            if out[i, j] < threshold:
               out[i, j] = 1e-10

    return out[:, 0] / VPlas
# 对数归一化和反归一化函数y[:, 0] / VPlas
def log_normalize(params):
    """对参数进行对数归一化"""
    return np.log(params)  # 确保对数变换只对正数操作
#   """对数归一化的反操作（指数恢复）"""
def exp_denormalize(log_params):
    """对数归一化的反操作（指数恢复）"""
    return np.exp(log_params)

call_count = 0
def total_cost(pars,dose_datas, time_points_train, concentration_data_train):
    global call_count
    call_count += 1  # 每次调用时增加计数器
    #print(f"Total cost 调用次数: {call_count}")
    # 打印输入参数
    #print(f"Parameters : {exp_denormalize(pars)}")
    total_cost = 0
    for idx in range(len(time_points_train)):
        DOSEdata = dose_datas[idx]
        Time_point=time_points_train[idx]
        result_df = MTX_PBPK(pars, Time_point, DOSEdata)
        observed_values = concentration_data_train[idx]
        print(f"组 {idx + 1} 的时间点: {Time_point},组 {idx + 1} 的预测值: {result_df}")
        #print(f"组 {idx + 1} 的观察值: {observed_values}")
        cost = np.sum((result_df - observed_values)**2)
        #print(f"组 {idx + 1} 的成本: {cost}")
        total_cost += cost

    #print(f"总成本: {total_cost}")
    return total_cost

# 定义先验分布
def prior_log_dist(pars):
    pars = np.array(pars)
    #print(f"prior_log_dist 调用, pars: {pars}, pars.shape: {pars.shape}")
    pars_data = exp_denormalize(pars)
    mean = exp_denormalize(pars)
    CV = 0.5  # 默认变异系数
    sd = mean * CV

    # 确保 mean 和 sd 是有效的数组
    if mean.shape[0] == 0:
        raise ValueError("mean array is empty")
    if sd.shape[0] == 0:
        raise ValueError("sd array is empty")

    # 计算百分位数
    lower_bounds = np.percentile(mean, 2.5, axis=0)
    upper_bounds = np.percentile(mean, 97.5, axis=0)

    # 确保百分位数的计算是有效的
    if lower_bounds.shape[0] == 0 or upper_bounds.shape[0] == 0:
        raise ValueError(" percentile array is empty")

    prior_pars = [truncnorm(loc=m, scale=s, a=(lower_bounds - m) / s, b=(upper_bounds - m) / s) for m, s in zip(mean, sd)]
    prior_sig2 = truncnorm(loc=0.5, scale=1.0, a=0.01, b=3.3)

    log_priors = [pm.logp(prior_dist, val) for prior_dist, val in zip(prior_pars, pars_data)]
    log_prior_sig2 = pm.logp(prior_sig2, 0.5)

    log_likelihood = sum(log_priors) + log_prior_sig2

    return -2 * log_likelihood

# 定义自定义似然函数
def log_likelihood(pars, sig2, time_points_train, dose_datas, concentration_data_train):
    total_cost_value = 0
    for i in range(len(time_points_train)):
        mu_pred = MTX_PBPK(pars, time_points_train[i], dose_datas[i])
        cost = np.sum((mu_pred - concentration_data_train[i])**2 / (2 * sig2**2))
        total_cost_value += cost
    return -total_cost_value
import matplotlib.pyplot as plt
import datetime
# import theano.tensor as tt
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
from scipy.optimize import minimize
import pymc3 as pm
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd

def derivshiv(y, t, parms, R, T_total):
    '''定义微分方程的函数，包含药物点滴输入'''
    
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    # 确保 input_rate 是标量
    input_rate = R if t <= T_total else 0

    ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate / VPlas
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]

    return ydot

# 药代动力学模型函数，用于MCMC优化
def MC_model(t, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {params}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        t, 
        args=(params, R, T_total), 
        rtol=1e-6,  # 放宽相对误差容忍度
        atol=1e-9,  # 放宽绝对误差容忍度
        h0=0.1     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    return y[:, 0] / VPlas

def total_cost(params, time_points_train, concentration_data_train):
    global call_count
    call_count += 1  # 每次调用时增加计数器
    #print(f"Total cost 调用次数: {call_count}")
    # 打印输入参数
    #print(f"Parameters : {exp_denormalize(pars)}")
    total_cost = 0
    for i in tqdm(range(len(time_points_train))):
    
        time_points = time_points_train[i]        
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen
        result_df = MC_model(time_points, D_total, T_total, *params) 
        observed_values = concentration_data_train[i]
        #print(f"组 {i + 1} 的时间点: {time_points},组 {i + 1} 的预测值: {result_df}")
        #print(f"组 {idx + 1} 的观察值: {observed_values}")
        cost = np.sum((result_df - observed_values)**2)
        #print(f"组 {i + 1} 的成本: {cost}")
        total_cost += cost
    print(f"总成本: {total_cost}")
    return total_cost
##############################--------modfit参数优化--------#################################################
#未优化的参数
pars = [init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"],
                 init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"],
             init_pars["Kreab"]]

param = pars
call_count = 0
#TODO:modfit拟合结果始终不对
#TODO:翻找之前的modfit
# ------ 添加验证代码的起始位置 ------
# 验证初始参数的目标函数值
initial_cost = total_cost(param, time_points_train, concentration_data_train)
print(f"Initial cost: {initial_cost}")

# 对 Kbile 增加 10% 扰动
perturbed_param = np.array(param).copy()  # 确保 param 是 numpy 数组
perturbed_param[3] *= 1.1  # Kbile 是第4个参数（索引3）
perturbed_cost = total_cost(perturbed_param, time_points_train, concentration_data_train)
print(f"Perturbed cost (Kbile +10%): {perturbed_cost}")
# ------ 添加验证代码的结束位置 ------

# 定义一个调试目标函数
# 开始计时
start_time = time.time()
# 使用 minimize 函数进行参数优化
bounds = [(0.01 * p, 10.0 * p) for p in param]
#options = {'disp': True, 'maxiter': 1000, 'ftol': 1e-5}
#def total_cost(params, time_points_train, concentration_data_train):
result = minimize(total_cost,  param, args = (time_points_train, concentration_data_train),
                      bounds=bounds,method = 'Nelder-Mead')# , options=options
# 结束计时total_cost(pars,dose_datas, time_points_train, concentration_data_train)
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"minimize 函数的运行时间为: {elapsed_time:.4f} 秒")
#初始优化参数
popt = result.x
print("优化结果消息:", result.message)
print("是否成功:", result.success)
print("最终目标函数值:", result.fun)

print(f"原始参数: \n{init_pars}")
print(f"优化参数: \n{popt}")

# 保存优化后的参数
with open('optimized_params0426.pkl', 'wb') as f:
    pickle.dump(popt, f)

print("优化参数已保存到 optimized_params0426.pkl")
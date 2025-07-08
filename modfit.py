import matplotlib.pyplot as plt
import datetime
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
from scipy.optimize import minimize
# import pymc3 as pm
# import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from ode_core import derivshiv,PK_model,FIT_model,log_normalize,exp_denormalize
# 获取当前日期
today_date = datetime.datetime.now().strftime('%Y-%m-%d') 

def total_cost(log_params, time_points_train, concentration_data_train):
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
        pars_linear = exp_denormalize(log_params)
        result_df = FIT_model(time_points, D_total, T_total, *pars_linear) 
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

#param = pars
log_pars = log_normalize(pars)
call_count = 0
# 定义一个调试目标函数
# 开始计时
start_time = time.time()
# 使用 minimize 函数进行参数优化
param_bounds_linear = [
    (0.10, 0.30),   # PRest
    (1.0,  3.5),    # PK
    (2.0,  6.0),    # PL
    (0.5,  5.0),    # Kbile (h^-1)
    (5.0,  25.0),   # GFR  (L h^-1)
    (0.40, 0.80),   # Free (fraction)
    (20.0, 2000.0), # Vmax_baso (mg h^-1)
    (5.0,  60.0),   # Km_baso  (mg L^-1)
    (0.01, 0.20),   # Kurine (h^-1)
    (0.00, 0.20)    # Kreab  (h^-1)
]  # ★★ 仅此列表被替换
bounds = [(np.log(lo), np.log(hi)) for lo, hi in param_bounds_linear]

result = minimize(total_cost,  
                  log_pars, 
                  args = (time_points_train, concentration_data_train),
                  method = 'Powell')#bounds=bounds, , options=options
# 结束计时total_cost(pars,dose_datas, time_points_train, concentration_data_train)
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"minimize 函数的运行时间为: {elapsed_time:.4f} 秒")
#初始优化参数
log_opt = result.x
popt = exp_denormalize(log_opt)

print("优化结果消息:", result.message)
print("是否成功:", result.success)
print("最终目标函数值:", result.fun)

print(f"原始参数: \n{init_pars}")
print(f"优化参数: \n{popt}")

# 保存优化后的参数
with open(f'saved_result/optimized_params{today_date}.pkl', 'wb') as f:
    pickle.dump(popt, f)

print("✔🌟优化参数已保存")
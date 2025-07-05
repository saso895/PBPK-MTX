import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
#import pymc3 as pm
#print(pm.__file__)
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
#dosing_time_train,rate_data_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from ode_core import derivshiv 

# --------------------------------------------------------------
# 1. PBPK 方程 已导入

# 药代动力学模型函数，用于参数拟合
#TODO:传参顺序不对，把新版本的微分方程和model拷贝过来
def pk_model(t, D_total, T_total, Duration,*param):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {param}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    # Specify time points to simulate
    Time=np.arange(0, Duration + 0.1, 0.1)
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        Time, 
        args=(param, R, T_total), 
        #method='BDF',
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    #return y[:, 0] / VPlas
    CA = y[:, 0] / VPlas
    results = np.column_stack((Time, CA))
    return results

##############################--------原始参数 + modfit参数的拟合结果--------#################################################
# 获取当前日期
today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错、
with tqdm(range(len(time_points_train))) as ybar:
    
    #with open('saved_result/optimized_params0528_Powell.pkl', 'rb') as f:
    #    params = pickle.load(f)
    # === ⬇️ 载入 0606 最优参数向量（完整 10 维）======================
    with open('207result\chain1_params.pkl', 'rb') as f:#saved_result/
        saved = pickle.load(f)              # 读回 dict
        saved = saved[:10] 
    #baseline = saved 
    #baseline = saved['baseline']            # numpy.ndarray (10,)
    if isinstance(saved, dict):
        # 通常你之前保存的是 {'baseline': ndarray, ...}
        baseline = np.asarray(saved.get('baseline',      # 首选键
                                        saved.get('params')))  # 备选键
    else:
        baseline = np.asarray(saved)  
    params   = baseline.tolist()            # 转成普通 list，以免 *params 时出 warning
    # ================================================================

    # 创建一个列表来存储每个病人的观测变量
    y_mu = []
    #y_obs = []
    for i in ybar:
    # for i in tqdm(range(len(time_points_train))):
        # pars.append(params)
        Duration= time_points_train[i][-1]
        time_points = time_points_train[i]
        concentration = concentration_data_train[i]+(10e-6)
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen
        
        # 计算预测浓度
        mu = pk_model(time_points, D_total, T_total,Duration, *params)
        y_mu.append(mu)  

#save_path =f'saved_result/SimuData_{today_date}.pkl' 
save_path =f'207result/chain1_0620_207.pkl' 
with open(save_path, 'wb') as f:
    pickle.dump(y_mu, f)

print("✔🌟预测结果已保存")

    

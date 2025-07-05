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
from ode_core import derivshiv ,PK_model

# --------------------------------------------------------------
# 1. PBPK 方程已导入
# 2. simu 函数已导入

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
        mu = PK_model(time_points, D_total, T_total,Duration, *params)
        y_mu.append(mu)  

#save_path =f'saved_result/SimuData_{today_date}.pkl' 
save_path =f'207result/chain1_0620_207.pkl' 
with open(save_path, 'wb') as f:
    pickle.dump(y_mu, f)

print("✔🌟预测结果已保存")

    

import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import pymc3 as pm
print(pm.__file__)
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
#dosing_time_train,rate_data_train
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
#today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错、
with tqdm(range(len(time_points_train))) as ybar:

    genetic_params = [
        0.0374, 
        4.68, 
        0.697, 
        0.33 , 
        13.63, 
        183.053, 
        183.053, 
        17.17, 
        0.5239, 
        0.935
    ]
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
        mu = pk_model(time_points, D_total, T_total,Duration, *genetic_params)
        y_mu.append(mu)  
with open('GA_simu0506.pkl', 'wb') as f:
    pickle.dump(y_mu, f)

print("GA预测结果已保存到 GA_simu0506.pkl")

    

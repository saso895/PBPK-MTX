import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
import numpy as np
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
#dosing_time_train,rate_data_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from ode_core import derivshiv ,PK_model
from init_param import (            # 原始生理参数
    init_pars,                      # ← 10-dim ndarray
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
)
# 获取当前日期
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

# --------------------------------------------------------------
# 1. PBPK 方程已导入
# 2. simu 函数已导入
# --------------------------------------------------------------
# 3. 灵活的参数导入函数
def load_parameters(source="init", file_path=None, idx=None):
    """
    Parameters
    ----------
    source : {"init", "modfit", "mcmc", "file"}
        选择参数来源
    file_path : str or None
        当 source 为 "file" / "modfit" / "mcmc" 时的 pkl 路径
    idx : int or None
        当 pkl 文件里包含多条链 (dict) 时，可指定取第几条
    Returns
    -------
    list
        10-维参数列表，可直接 *params 解包
    """
    if source == "init":
        return list(init_pars.values())[:10]

    if file_path is None:
        raise ValueError("source=%s 需要提供 file_path" % source)

    with open(file_path, "rb") as f:
        loaded = pickle.load(f)

    # ----- modfit: 通常是 ndarray (10,) 或 dict{'baseline':...}
    if source == "modfit":
        if isinstance(loaded, dict):
            loaded = loaded.get("baseline", loaded.get("params"))
        return np.asarray(loaded)[:10].tolist()

    # ----- mcmc: 通常是 ndarray(11,) 或 chain_dict
    if source == "mcmc":
        if isinstance(loaded, dict):         # 多链 dict
            if idx is None:
                idx = 1
            key = f"chain{idx}_params" if f"chain{idx}_params" in loaded else list(loaded.keys())[0]
            loaded = loaded[key]
        return np.asarray(loaded)[:10].tolist()

    # ----- 任意手动文件
    if source == "file":
        return np.asarray(loaded)[:10].tolist()

    raise ValueError("未知 source：%s" % source)
# --------------------------------------------------------------
# 4. 浓度的拟合结果保存
# 定义保存路径
save_dir = f'saved_result'
with tqdm(range(len(time_points_train))) as ybar:
    
### >>> 选择参数来源
    PARAM_SOURCE = "modfit"          # {"init","modfit","mcmc","file"}
    PARAM_FILE   = "saved_result/modfit02_params2025-07-11.pkl"   # ← 自行修改路径
    CHAIN_IDX    = 1               # mcmc 多链时选第几链  
    params = load_parameters(PARAM_SOURCE,PARAM_FILE)   
### >>> 预测浓度
    # 创建一个列表来存储每个病人的观测变量
    y_mu = []
    #y_obs = []
    for i in ybar:
        Duration= time_points_train[i][-1]
        time_points = time_points_train[i]
        concentration = concentration_data_train[i]+(10e-6)
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen        
        # 计算预测浓度
        mu = PK_model(time_points, D_total, T_total, Duration, *params)
        y_mu.append(mu)
        if i == 0:                                    # 只打印第一个病例就够对比
            print(f"Cmax (case 1) = {mu[:,1].max():.3f} mg/L")  

save_path =f'{save_dir}/simu02_{PARAM_SOURCE}_{today_date}.pkl' 
with open(save_path, 'wb') as f:
    pickle.dump(y_mu, f)

print("✔🌟预测结果已保存")

    

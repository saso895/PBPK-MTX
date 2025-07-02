import pandas as pd
import numpy as np
from scipy.optimize import minimize
from MTX_PBPK1122 import MTX_PBPK, total_cost,log_normalize, exp_denormalize
from pars1111 import params
from datetime import datetime
import matplotlib.pyplot as plt
from init_data1029 import data, arm_data_train,time_points_train, time_points_test, concentration_data_train, concentration_data_test, rate_data_train, rate_data_test, dosing_time_train, dosing_time_test
import pickle
import time
import pymc as pm
from scipy.stats import truncnorm, uniform, norm
import os
from joblib import Parallel, delayed

#未优化的参数
init_pars = [params["PRest"], params["PK"], params["PL"], params["Kbile"], params["GFR"],
                 params["Free"], params["Vmax_baso"], params["Km_baso"], params["Kurine"],
             params["Kreab"]]

# 对 initial_guess 进行对数归一化
initial_guess_normalized = log_normalize(init_pars)

# 对训练集进行拟合
optimized_params = []
# 提前计算并准备好 durations 和 dose_datas
durations = [data.loc[data['Arm'] == arm, 'Time'].max().item() for arm in arm_data_train]
dose_datas = [pd.DataFrame({'time': [dosing_time_train[idx]], 'Rate': [rate_data_train[idx]]}) for idx in range(len(arm_data_train))]

param = initial_guess_normalized
call_count = 0
# 定义一个调试目标函数
# 开始计时
start_time = time.time()
# 使用 minimize 函数进行参数优化
bounds = [(np.log(0.01), np.log(10.0)) for _ in range(len(initial_guess_normalized))]
options = {'disp': True, 'maxiter': 1000, 'ftol': 1e-5}
result = minimize(total_cost,  param, args = (dose_datas , time_points_train, concentration_data_train),
                      method = 'L-BFGS-B', bounds=bounds, options=options)
# 结束计时total_cost(pars,dose_datas, time_points_train, concentration_data_train)
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"minimize 函数的运行时间为: {elapsed_time:.4f} 秒")
#初始优化参数
popt = exp_denormalize(result.x)

print(f"原始参数: \n{init_pars}")
print(f"优化参数: \n{popt}")

# 保存优化后的参数
with open('optimized_params.pkl', 'wb') as f:
    pickle.dump(popt, f)

print("优化参数已保存到 optimized_params.pkl")
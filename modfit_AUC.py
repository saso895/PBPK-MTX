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

# === MOD BEGIN ❶ : 计算 ε = LLOQ/2 ====================================
# 1) 用于将观测值中的 0 / BLQ 替换为 ε
# 2) 用于对数残差：log(pred + ε) - log(obs + ε)
_positive_vals = np.concatenate([arr[arr > 0] for arr in concentration_data_train])
if _positive_vals.size == 0:
    raise ValueError("训练数据全部为 0，无法确定 LLOQ")
LLOQ = _positive_vals.min()
EPS = LLOQ / 2.0          # 例如最小 0.00317 → EPS ≈ 0.0016
print(f"▶  Using ε = LLOQ/2 = {EPS:.4g} mg/L  for log-SSE")
# === MOD END ❶ ========================================================


def total_cost(log_params, time_points_train, concentration_data_train):
    global call_count
    call_count += 1  # 每次调用时增加计数器
    #print(f"Total cost 调用次数: {call_count}")
    # 打印输入参数
    #print(f"Parameters : {exp_denormalize(pars)}")
    total_auc_sse = 0.0  
    #total_cost = 0
    pars_linear = exp_denormalize(log_params)
    for i in tqdm(range(len(time_points_train))):
    
        time_points = time_points_train[i]        
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        
        result_df = FIT_model(time_points, dose, timelen, *pars_linear) 
        # ------ 观测值预处理 --------------------------------------------
        obs_raw = concentration_data_train[i]
        obs_use = np.where(obs_raw <= 0, EPS, obs_raw)   # 0 → ε
                # --- 计算 AUC（梯形法）----------------------------------------
        auc_pred = np.trapz(result_df, time_points)
        auc_obs  = np.trapz(obs_use, time_points)
        #observed_values = concentration_data_train[i]
        #print(f"组 {i + 1} 的时间点: {time_points},组 {i + 1} 的预测值: {result_df}")
        #print(f"组 {idx + 1} 的观察值: {observed_values}")    
        # ------ 对数残差 ------------------------------------------------
        log_res_sq = (np.log(auc_pred + EPS) - np.log(auc_obs)) ** 2
                # === MOD BEGIN ❷ : 高浓 ↑权重 / 低浓 ↓权重 =============
        # 以 1 mg·L⁻¹ 为阈值：>1 → 2.0，≤1 → 0.5

        #w = 0.3 + 2.7 * (obs_use / 1.0)**0.58    
        total_auc_sse += log_res_sq

    print(f"对数总成本: {total_auc_sse}")
    return total_auc_sse
##############################--------modfit参数优化--------#################################################
#未优化的参数
pars = [init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"],
                 init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"],
             init_pars["Kreab"]]
param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab"
]
pars_linear = [init_pars[p] for p in param_names]
#param = pars
log_pars = log_normalize(pars)
call_count = 0
# 定义一个调试目标函数
# 开始计时
start_time = time.time()
# 使用 minimize 函数进行参数优化
pk0   = init_pars["PK"]
pl0   = init_pars["PL"]
kur0  = init_pars["Kurine"]
vmax0 = init_pars["Vmax_baso"]
kur0  = init_pars["Kurine"]
pr0   = init_pars["PRest"]

param_bounds_linear = [
    (0.15,  0.30),   # PRest
    # (pr0  * 0.3, pr0  * 5.0),   # PRest   ← 放宽
    # (pk0  * 0.2, pk0  * 8.0),  # PK      ← 放宽
    # (pl0  * 0.2, pl0  * 8.0),  # PL      ← 放宽
     (0.100,  5.00),    # PK
     (0.100,  5.00),    # PL
    (0.50,  5.00),    # Kbile (h^-1)
    (5.00,  25.0),   # GFR  (L h^-1)
    (0.45,  0.76),   # Free (fraction)
    #(vmax0 * 0.1, vmax0 * 10.0),# Vmax_baso ← 放宽
    (20.0,  600.0), # Vmax_baso (mg h^-1)
    (5.00,  300.0),   # Km_baso  (mg L^-1)
    (0.02,  0.25),   # Kurine (h^-1)
    #(kur0 * 0.2, kur0 * 8.0),  # Kurine  ← 放宽
    (0.00,  0.20)    # Kreab  (h^-1)
]  # ★★ 仅此列表被替换
bounds = [(np.log(lo), np.log(hi)) for lo, hi in param_bounds_linear]

result = minimize(total_cost,  
                  log_pars, 
                  args = (time_points_train, concentration_data_train),
                  method = 'Powell',
                  bounds=bounds             # ★★ <-- 把这行加回来
                  )#bounds=bounds, , options=options

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
print("\n┌──────────┬────────────┬────────────┐")
print("│ Parameter│  Initial   │  Optimized │")
print("├──────────┼────────────┼────────────┤")
for n, v0, v1 in zip(param_names, pars_linear, popt):
    print(f"│ {n:<9}│ {v0:>10.4g} │ {v1:>10.4g} │")
print("└──────────┴────────────┴────────────┘\n")
# print(f"原始参数: \n{init_pars}")
# print(f"优化参数: \n{popt}")

# 保存优化后的参数
with open(f'saved_result/modfit_auc_params{today_date}.pkl', 'wb') as f:
    pickle.dump(popt, f)

print("✔🌟优化参数已保存")
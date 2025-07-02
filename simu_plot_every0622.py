import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time,datetime
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

#==========读入模拟数据
with open('207result\chain1_0619_207.pkl', 'rb') as f:
    y_chain1=pickle.load( f)
# === 计算对数坐标用的全局最小、最大浓度（排除0）===
with tqdm(range(len(time_points_train))) as pbar:
# ============ 🆕 逐病人 GOF（chain-1） ============
    num_patients = len(time_points_train)
    ncols = 3                                         # 每行 3 张子图，可自行调整
    nrows = (num_patients + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                            figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    global_min, global_max = np.inf, -np.inf
    pred_cache = []
    all_pred_vals = []
    all_obs_vals  = []
    for i in range(num_patients):
        # t_obs = time_points_train[i]
        # c_obs = concentration_data_train[i]
                # ✅ 改成 ↓↓↓ －－－－－－－－－－－－－－－－－－
                       # 只要 t>0 的三个时点
        t_obs = np.asarray(time_points_train[i]).astype(float).ravel()
        c_obs = np.asarray(concentration_data_train[i]).astype(float).ravel()
        mask = (t_obs > 0)
        t_obs, c_obs = t_obs[mask], c_obs[mask]


        #curve = y_chain1[i]                           # [[time, conc], …]
                # ------- 预测值 (同样 3 点) -------
        curve = np.asarray(y_chain1[i], dtype=float)
        if curve.ndim == 2:                   # [[t, C], ...]
            c_pred = np.interp(t_obs, curve[:, 0], curve[:, 1])
        else:                                 # 已经是一一对应
            c_pred = curve.ravel()[mask]
        # 线性插值到观测点
        c_pred = np.interp(t_obs, curve[:, 0], curve[:, 1])
        pred_cache.append((c_pred, c_obs))
        all_pred_vals.append(c_pred)
        all_obs_vals.append(c_obs)
        #$pred_cache.append((c_pred, c_obs))
        # 拼成一维
    all_pred_vals = np.concatenate(all_pred_vals)
    all_obs_vals  = np.concatenate(all_obs_vals)
    glob_min = min(all_pred_vals.min(), all_obs_vals.min())
    glob_max = max(all_pred_vals.max(), all_obs_vals.max())
    #lobal_max = max(global_max, c_pred.max(), c_obs.max())
    pad_low  = 0.5   # 对应 10^(-0.5) ≈ ×0.32
    pad_high = 0.5   # 对应 10^(+0.5) ≈ ×3.16
    x_min = glob_min * (10**-pad_low)
    x_max = glob_max * (10** pad_high)
    for i, ax in enumerate(axes):
        if i >= num_patients:
            ax.axis('off')
            continue

        c_pred, c_obs = pred_cache[i]
        r2   = r2_score(c_obs, c_pred)
        rmse = np.sqrt(mean_squared_error(c_obs, c_pred))
        print(f"Patient {i+1}: pred={c_pred}, obs={c_obs}") 

        ax.scatter(c_pred, c_obs, alpha=0.7)
        #ax.plot([global_min, global_max], [global_min, global_max],
        #        'r--', linewidth=1)

        ax.set_xlabel('Predicted Conc. (mg/L)log')
        ax.set_ylabel('Observed Conc. (mg/L)log')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Patient {i+1}\n$R^2={r2:.3f}$  RMSE={rmse:.2f}$')
        ax.scatter(c_pred, c_obs, s=50, alpha=0.7,
           edgecolors="k", linewidths=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.plot([x_min, x_max], [x_min, x_max], 'r--', lw=1)   # 全局对角线
        # ax.set_xlim(global_min, global_max)
        # ax.set_ylim(global_min, global_max)
        ax.grid(True, linestyle=':')
        


    fig.tight_layout()

    os.makedirs('saved_result', exist_ok=True)
    fig.savefig('207result/GOF_chain1_by_patient.svg', format='svg')
    plt.show()
    

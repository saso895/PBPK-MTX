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
from sklearn.metrics import mean_absolute_error
today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# ====================== 一键切换服务器 ======================
SERVER_ID  = 207            # ← 只改这里：207 或 209
BASE_DIR   = f"{SERVER_ID}result"
os.makedirs(BASE_DIR, exist_ok=True)
# ==========================================================

def relmae_grade(relmae):
    if relmae <= 0.10:
        return "4"
    elif relmae <= 0.20:
        return "3"
    elif relmae <= 0.40:
        return "2"
    else:
        return "1"

color_dict = {"4": "green", "3": "dodgerblue",
              "2": "orange", "1": "red"}


#==========读入模拟数据
with open(f'{BASE_DIR}\chain1_0619_207.pkl', 'rb') as f:
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
    pad_low  = 0.5   # 对应 10^(-0.5) ≈ ×0.32
    pad_high = 0.5   # 对应 10^(+0.5) ≈ ×3.16
    x_min = glob_min * (10**-pad_low)
    x_max = glob_max * (10** pad_high)
    all_mae = []
    all_rmse = []
    all_mean_obs, all_relmae, all_grade = [], [], []
    all_afe = []
    for i, ax in enumerate(axes):
        if i >= num_patients:
            ax.axis('off')
            continue

        c_pred, c_obs = pred_cache[i]
        #r2   = r2_score(c_obs, c_pred)
        rmse = np.sqrt(mean_squared_error(c_obs, c_pred))
        mae = mean_absolute_error(c_obs, c_pred)
        all_mae.append(mae)
        all_rmse.append(rmse)
        mean_obs = np.mean(c_obs)
        relmae   = mae / mean_obs if mean_obs > 0 else np.nan
        grade    = relmae_grade(relmae)

        all_mean_obs.append(mean_obs)
        all_relmae.append(relmae)
        all_grade.append(grade)

        print(f"Patient {i+1}: pred={c_pred}, obs={c_obs}") 

        ax.scatter(c_pred, c_obs, alpha=0.7)
        ax.set_xlabel('Predicted Conc. (mg/L)log')
        ax.set_ylabel('Observed Conc. (mg/L)log')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend([f"MAE={mae:.3f}\nRMSE={rmse:.3f}\n{grade}"], loc="upper left", frameon=True)
        
        ax.set_title(f'Patient {i+1}')
        ax.scatter(c_pred, c_obs, s=50, alpha=0.7,
           edgecolors="k", linewidths=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.plot([x_min, x_max], [x_min, x_max], 'r--', lw=1)   # 全局对角线
        ax.plot([x_min, x_max], [x_min*0.5, x_max*0.5], 'r--', lw=1, alpha=0.6)  # 0.5×
        ax.plot([x_min, x_max], [x_min*2.0,  x_max*2.0 ], 'r--', lw=1, alpha=0.6)  # 2×
        ax.grid(True, linestyle=':')
        # ③ —— （可选）计算 Fold-error 与 AFE 并写入汇总表
        ratio        = c_pred / c_obs                       # Pred/Obs
        fold_error   = ratio                                # 每个观测点
        afe_patient  = np.exp(np.mean(np.abs(np.log(ratio))))  # AFE (几何平均)
        # …在 all_* 列表中同步收集
        all_afe.append(afe_patient)
    df_gof = pd.DataFrame({
    'Patient'  : [f'Patient_{i+1}' for i in range(num_patients)],
    #'MAE'      : all_mae,
    'RMSE'     : all_rmse,
    'MeanObs'  : all_mean_obs,
    #'RelMAE'   : all_relmae,
    'AFE'     : all_afe,              # 新列
    'Grade'    : all_grade
})   

    # 打印
    print(df_gof)
    # 或导出Excel
    os.makedirs(BASE_DIR, exist_ok=True)
    df_gof.to_excel(f'{BASE_DIR}/gof_patient_table{today_date}.xlsx', index=False)
    # === 汇总统计 ===
    print("\n—— 分级统计 ——")
    print(df_gof['Grade'].value_counts())
    print("全体MAE均值：", df_gof['MAE'].mean())
    print("全体RMSE均值：", df_gof['RMSE'].mean())

    fig.tight_layout()
    
    fig.savefig(f'{BASE_DIR}/GOF_chain1_by_patient_RMSE{today_date}.svg', format='svg')
    plt.show()
    
# === 颜色柱状图 ===
bar_colors = df_gof['Grade'].map(color_dict)
fig_bar, ax_bar = plt.subplots(figsize=(15,5))
df_gof['RelMAE'].plot(kind='bar', color=bar_colors, ax=ax_bar)

ax_bar.set_ylabel('Relative MAE')
ax_bar.set_xlabel('Patient')
ax_bar.set_title('各病人相对 MAE（颜色按分级）')
ax_bar.set_xticklabels(df_gof['Patient'], rotation=90, fontsize=7)
ax_bar.axhline(0.10, ls='--', color='gray',   label='10% 优秀')
ax_bar.axhline(0.20, ls='--', color='dodgerblue', label='20% 良好')
ax_bar.axhline(0.40, ls='--', color='orange', label='40% 一般')
ax_bar.legend()
fig_bar.tight_layout()
fig_bar.savefig(f'{BASE_DIR}/relmae_barplot{today_date}.png', dpi=300)
plt.show()

########----------“较差”病人规律
bad_df = df_gof[df_gof['Grade']=='较差']
print("\n—— 拟合较差的病例 ——")
print(bad_df[['Patient','MAE','RelMAE','MeanObs']])

# 简单相关性检查
corr = df_gof[['RelMAE','MeanObs']].corr().iloc[0,1]
print(f"\nRelMAE 与 MeanObs 相关系数: {corr:.3f}")

# 可视化散点
fig_sc, ax_sc = plt.subplots()
ax_sc.scatter(df_gof['MeanObs'], df_gof['RelMAE'],
              c=df_gof['Grade'].map(color_dict), alpha=0.8)
ax_sc.set_xscale('log')
ax_sc.set_xlabel('Mean Observed Conc. (log)')
ax_sc.set_ylabel('Relative MAE')
ax_sc.set_title('RelMAE vs MeanObs')
fig_sc.tight_layout()
fig_sc.savefig(f'{BASE_DIR}/relmae_vs_meanobs{today_date}.png', dpi=300)
plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
局部灵敏度 + 共线指数 (类似 Brun et al., 2001)
------------------------------------------------
1. 对 10 个 PBPK 参数在基线处做有限差分
2. 计算加权灵敏度矩阵 S
3. 输出:
   • l_msqr.csv      (RMS 灵敏度排名)
   • collinearity.csv(单参数加入时的共线指数)
   • bar_hm.png      (柱状 + 热图可视化)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy.linalg import svd
from init_param import init_pars
from init_data_point4 import time_points_train, input_dose_train, inject_timelen_train
from modmcmc0508 import derivshiv, VPlas

# ---------- 基线 --------------
theta0 = np.array([
    init_pars['PRest'], init_pars['PK'], init_pars['PL'], init_pars['Kbile'],
    init_pars['GFR'], init_pars['Free'], init_pars['Vmax_baso'],
    init_pars['Km_baso'], init_pars['Kurine'], init_pars['Kreab']
])
names = np.array(['PRest','PK','PL','Kbile','GFR','Free',
                  'Vmax_baso','Km_baso','Kurine','Kreab'])
d      = len(theta0)

# ---------- 仿真输出 (浓度时序拼接成一长向量) ----------
def model_output(theta):
    # 血浆浓度, 拼成 1D array (样本点总数 ≈ 73)
    outs = []
    for t_pts, dose, tlen in zip(time_points_train,
                                 input_dose_train,
                                 inject_timelen_train):
        R = dose / tlen
        y = odeint(derivshiv, np.zeros(7), t_pts,
                   args=(theta, R, tlen),
                   rtol=1e-4, atol=1e-6, h0=0.1)
        outs.append(y[:,0] / VPlas)
    return np.concatenate(outs)

y0 = model_output(theta0)
N  = y0.size

# ---------- 定义量纲尺度 sc_yk ------------
# 这里简单用观测值的标准差; 若有实验误差请替换
obs_concat = np.concatenate([x for x in time_points_train])*0 + 1  # dummy
sc = np.std(y0) if np.std(y0)>0 else 1.0

# ---------- 有限差分灵敏度矩阵 ----------
S = np.zeros((N, d))
rel_step = 1e-3
for j in range(d):
    theta_p = theta0.copy()
    theta_p[j] *= (1+rel_step)
    dy = model_output(theta_p) - y0
    S[:, j] = dy / (theta0[j]*rel_step) / sc   # 归一化

# ---------- RMS 灵敏度 l_msqr ----------
l_msqr = np.sqrt((S**2).mean(axis=0))
pd.DataFrame({'param': names, 'l_msqr': l_msqr}) \
  .sort_values('l_msqr', ascending=False) \
  .to_csv('l_msqr.csv', index=False)

# ---------- 共线指数: baseline + 每个参数单独加入 ----------
u, svals, vt = svd(S, full_matrices=False)
gamma_full = svals[0]/svals[-1]
print("gamma_full =", gamma_full)
gamma_single = []
for j in range(d):
    idx = [j]  # 单独考察加入第 j 个
    Sj = S[:, idx]
    _, s, _ = svd(Sj, full_matrices=False)
    gamma_single.append(s[0]/s[-1] if len(s)>1 else 1.0)

pd.DataFrame({'param': names, 'collinearity': gamma_single}) \
  .to_csv('collinearity.csv', index=False)

# ---------- 可视化 ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
rank = np.argsort(l_msqr)[::-1]
plt.bar(np.arange(d), l_msqr[rank])
plt.xticks(np.arange(d), names[rank], rotation=60)
plt.ylabel('l_msqr (RMS sensitivity)')

plt.subplot(1,2,2)
plt.bar(np.arange(d), np.array(gamma_single)[rank])
plt.xticks(np.arange(d), names[rank], rotation=60)
plt.ylabel('Collinearity index γ')

plt.suptitle('Local sensitivity & collinearity (PBPK)')
plt.tight_layout()
plt.savefig('bar_hm.png', dpi=300)

print("✅ 局部灵敏度分析完成 → l_msqr.csv, collinearity.csv, bar_hm.png")
# === 共线指数 for 任意子集 ==============================================
from itertools import combinations

def collinearity_for_subsets(S, names, subset_sizes=tuple(range(2, 11))):
    """
    返回指定大小的全部参数组合的共线指数 γ，
    并把结果写入 CSV：collinearity_k*.csv
    """
    for k in subset_sizes:
        rows = []
        for combo in combinations(range(S.shape[1]), k):
            Sj = S[:, combo]               # 取子矩阵
            _, s, _ = np.linalg.svd(Sj, full_matrices=False)
            gamma = s[0]/s[-1] if len(s) > 1 else 1.0
            rows.append({
                'subset': ','.join(names[list(combo)]),
                'gamma' : gamma
            })
        df = pd.DataFrame(rows).sort_values('gamma', ascending=False)
        fname = f'saved_result/collinearity_k{k}.csv'
        df.to_csv(fname, index=False)
        print(f'✓ saved {fname}  (worst γ = {df.gamma.iloc[0]:.2f})')

# ---------- 调用示例 ----------
collinearity_for_subsets(S, names, subset_sizes=tuple(range(2, 11)))
print("✅ 所有组合局部灵敏度分析完成")

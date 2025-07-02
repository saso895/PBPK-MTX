#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global sensitivity analysis for modmcmc0508:
1. Morris screening on all 10 parameters
2. Sobol indices on the top-k (k = 5) most influential ones
Outputs:
    ├─ morris_mu_star.csv
    └─ sobol_S1_ST.csv
说明脚本做 两步全局敏感性：① Morris “萤火虫法” 粗筛；② 在前 k = 5 个高敏感参数上算 Sobol 方差分解。
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tqdm import tqdm
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.sample import saltelli
from SALib.analyze import sobol

from init_param import init_pars               # baseline parameter dict
from modmcmc0508 import derivshiv, VPlas       # ODE function & plasma volume

# ---------------------------------------------------------------------
# 0.  baseline vector & sampling bounds
#把 10 维基线参数做成 numpy 向量；
# 这就是后面 SALib 采样的超立方体。
# ---------------------------------------------------------------------
baseline = np.array([
    init_pars['PRest'], init_pars['PK'], init_pars['PL'], init_pars['Kbile'],
    init_pars['GFR'],   init_pars['Free'], init_pars['Vmax_baso'],
    init_pars['Km_baso'], init_pars['Kurine'], init_pars['Kreab']
])

# one log-unit up & down (feel free to tighten if ODE 爆炸)
#再构造 “上下各 1 log-unit” 的矩形采样空间 bounds。
bounds = np.vstack([baseline / 10.0, baseline * 10.0]).T

'''	核心包装函数

'''
problem = {
    'num_vars': 10,
    'names': ['PRest', 'PK', 'PL', 'Kbile', 'GFR', 'Free',
              'Vmax_baso', 'Km_baso', 'Kurine', 'Kreab'],
    'bounds': bounds.tolist()
}

# ---------------------------------------------------------------------
# 1. model wrapper — returns 0-24 h AUC
# ---------------------------------------------------------------------
def run_model(param_vector, active_idx=None, t_end=24):
    """
    Parameters
    ----------
    param_vector : array-like
        • full 10-D vector  —— if active_idx is None  
        • sub-vector       —— if active_idx is list/array of indices
    active_idx   : iterable or None
        Indices being overwritten in `baseline`.
    1. 若 active_idx is None：param_vector 已含全部 10 维。
    2. 否则把 “子向量” 写回 baseline 得到完整 10 维。
    3. 传给 odeint 解 0–24 h 浓度曲线，积分算 AUC<sub>0-24h</sub> 作为敏感性指标 Y。
    """
    # assemble complete 10-D param set
    if active_idx is None:
        full = np.asarray(param_vector, dtype=float)
    else:
        full = baseline.copy()
        full[list(active_idx)] = param_vector

    (PRest, PK, PL, Kbile, GFR,
     Free, Vmax_baso, Km_baso, Kurine, Kreab) = full

    # dummy infusion (mg / h) — adjust to your protocol if needed
    R = 1000.0

    y0 = np.zeros(7)
    t  = np.linspace(0, t_end, int(t_end*10) + 1)     # 0.1 h step
    y  = odeint(derivshiv, y0, t, args=(full, R, 1))
    CA = y[:, 0] / VPlas

    return np.trapz(CA, t)                            # AUC_0-24 h

# ---------------------------------------------------------------------
# 2. Morris screening
# ---------------------------------------------------------------------
'''① morris_sample.sample(problem, N) 生成 (N × (d + 1)) 轨迹 —— 每条轨迹只改动一个维度；
② 对每个参数算 μ★（平均绝对梯度）和 σ（方差）；
③ 保存 csv。
算法思想：用局部一阶增量Δ𝑖=𝑓(𝑥+Δ𝑒𝑖)−𝑓(𝑥)ΔΔ=Δf(x+Δei​)−f(x)​
近似灵敏度，再对随机多点求均值 → 低成本发现“谁动得最多”。
'''
N_MORRIS = 1000
print(f"\n🔍  Morris sampling ({N_MORRIS} trajectories × 10 dims)…")
X = morris_sample.sample(problem, N_MORRIS, optimal_trajectories=10)

Y = np.array([run_model(x) for x in tqdm(X, desc='Morris runs')])
Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=True)

mu_star = Si['mu_star']
pd.DataFrame({
    'parameter': problem['names'],
    'mu_star' : mu_star
}).to_csv('morris_mu_star.csv', index=False)

#取 μ★ 最大的 5 维，下标记入 top_idx。
top_idx   = np.argsort(mu_star)[::-1][:5]             # top-5 drivers
top_names = [problem['names'][i] for i in top_idx]
print("\nTop drivers by Morris:", list(zip(top_names, mu_star[top_idx])))

# ---------------------------------------------------------------------
# 3. Sobol on the most influential parameters
# ---------------------------------------------------------------------
'''① 构造子问题只有 top-k 维；
② saltelli.sample(..., calc_second_order=False) 生成 Saltelli 序列，具有2𝑑𝑁2dN 点，能一次性估 S1/ST；
③ 仍调用 run_model（但只改子向量），得到 Y₂；
④ sobol.analyze 输出：
 • S1 一阶方差贡献 
'''
sub_problem = {
    'num_vars': len(top_idx),
    'names'   : top_names,
    'bounds'  : bounds[top_idx].tolist()
}
N_SOBOL = 2048
print(f"\n🔍  Sobol sampling ({N_SOBOL} × {len(top_idx)} dims)…")
X2 = saltelli.sample(sub_problem, N_SOBOL, calc_second_order=False)

Y2 = np.array([run_model(vec, active_idx=top_idx)
               for vec in tqdm(X2, desc='Sobol runs')])

Si2 = sobol.analyze(sub_problem, Y2, 
                    calc_second_order=False,
                    print_to_console=True)

pd.DataFrame({
    'parameter': sub_problem['names'],
    'S1': Si2['S1'],
    'ST': Si2['ST']
}).to_csv('sobol_S1_ST.csv', index=False)

print("\n✅  Sensitivity analysis finished — results saved as:")
print("    • morris_mu_star.csv")
print("    • sobol_S1_ST.csv")

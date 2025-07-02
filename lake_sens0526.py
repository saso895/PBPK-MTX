# -*- coding: utf-8 -*-
"""
lake_sens02_plus.py —— 在一次敏感性+共线性分析基础上，
                      计算 Table-8 风格的参数标准误差 & 相关系数
-------------------------------------------------------------------
新增 Step-5:
  · 选定一个参数子集 (默认取全局灵敏度最高的 7 个)
  · 构造“加权雅可比” J_w  (与 WSS 中的加权方式一致)
  · 用 (J_w^T J_w)^-1 估计协方差矩阵，给出:
      - 相对标准误差 (%)
      - 参数间相关系数 ρ_jk
  · 结果打印 + 写入 table8_stats.csv
-------------------------------------------------------------------
其余 Step-1～Step-4 与原文件完全一致。
"""

import itertools, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg as la                     # 解决 la
from scipy.optimize import least_squares      # 解决 least_squares
from numpy.random import default_rng  
import numpy as np 
import os
from tqdm import tqdm
import pickle
import datetime

today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# --------------------------------------------------------------
# 0. 载入固定/可调参数 & 病人给药数据
# --------------------------------------------------------------
from init_param import (
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
    PRest, PK, PL, Kbile, GFR, Free,
    Vmax_baso, Km_baso, Kurine, Kreab
)
from init_data_point4 import (
    time_points_train as time_groups,
    input_dose_train  as dose_groups,
    inject_timelen_train as tinf_groups,
    concentration_data_train as conc_groups,
)

param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free",  "Vmax_baso", "Km_baso", "Kurine", "Kreab",
]
baseline = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
], dtype=float)

# --------------------------------------------------------------
# 1. PBPK 方程 & 模拟函数
# --------------------------------------------------------------
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    inp = R if t <= T_total else 0
    dy  = np.zeros(7)
    dy[0] = (
        QRest * y[3] / VRest / PRest
        + QK * y[2] / VK / PK
        + QL * y[1] / VL / PL
        - QPlas * y[0] / VPlas
        + Kreab * y[4]
        + inp / VPlas
    )
    dy[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    dy[2] = (
        QK * (y[0] / VPlas - y[2] / VK / PK)
        - y[0] / VPlas * GFR * Free
        - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    )
    dy[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    dy[4] = (
        y[0] / VPlas * GFR * Free
        + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
        - y[4] * Kurine
        - Kreab * y[4]
    )
    dy[5] = Kurine * y[4]
    dy[6] = Kbile * y[1]
    return dy

def FIT_model(t, dose, tinf, *params):
    R = dose / tinf
    y0 = np.zeros(7)
    sol = odeint(
        derivshiv, y0, t,
        args=(params, R, tinf),
        rtol=1e-6, atol=1e-9, h0=0.1,
    )
    return sol[:, 0] / VPlas   # 血浆浓度

# --------------------------------------------------------------
# 2. Step-1 —— 全局灵敏度
#使用文章中公式4-6计算
# --------------------------------------------------------------
delta_rel = 0.028
sl_vectors, S_global = [], []

base_curve, weights = [], []
for t, d, tinf in zip(time_groups, dose_groups, tinf_groups):
    y = FIT_model(t, d, tinf, *baseline)
    base_curve.extend(y)         #原始参数拟合浓度拼接
#    w = np.ones_like(y);         # 第二点权重×2（示例）
#    if len(y) >= 2: w[1] = 10.0
#    weights.extend(w)

#base_curve = np.array(base_curve)
# 拼接观测值 conc_groups 为一维数组
y_obs = np.concatenate(conc_groups)       # 实测浓度
y_pred = np.array(base_curve)             # 预测浓度（已提前生成）

# 计算非加权标准差（残差平方和均值开根号）
sy = np.sqrt(np.mean((y_obs - y_pred)**2))

for idx, (pname, theta0) in enumerate(zip(param_names, baseline)):
    dtheta = theta0 * delta_rel if theta0 != 0 else 1e-6
    up, down = baseline.copy(), baseline.copy()
    up[idx] += dtheta; down[idx] -= dtheta
    diff = np.concatenate([                         #使用中央查分公式求偏导
        FIT_model(t, d, tinf, *up) - FIT_model(t, d, tinf, *down)
        for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
    ]) / (2*dtheta)
    sl = (dtheta/sy) * (diff )                      #文中公式（4）
    sl_vectors.append(sl)                           #文中公式（5）
    S_global.append(math.sqrt(np.mean(sl**2)))      #文中公式（6）

order = np.argsort(S_global)[::-1]
print("\n=== Step-1  全局灵敏度排序 ===")
for rk, idx in enumerate(order, 1):
    print(f"{rk:2}. {param_names[idx]:<12}  S = {S_global[idx]:.3e}")

# --------------------------------------------------------------
# 3. Step-2 —— 共线性 γ（与你原脚本相同02）
# --------------------------------------------------------------
unit_vecs = [v/np.linalg.norm(v) for v in sl_vectors]#公式7

def gamma(idxs):
    S = np.column_stack([unit_vecs[i] for i in idxs])#公式7的小s构造出大S
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)#公式8根号里的内容
    return 1/math.sqrt(lam_min)#公式8最终结果

# γ 包络 (min/max) 

#size_list, g_low, g_high = [], [], []
plt.figure()
for k in range(1, len(param_names) + 1):
    g_vals = [gamma(c) for c in itertools.combinations(range(len(param_names)), k)]
    #size_list.append(k)
    #g_low.append(min(g_vals)); g_high.append(max(g_vals))
    xvals = [k] * len(g_vals)
    plt.plot(xvals, g_vals, 'ko', alpha=0.6, markersize=4)  # 黑色圆点
# 画斜线（如要标注灵敏度Top-n子集的γ）
# 例：用灵敏度最高7个参数顺次累计形成的子集
special_idxs = order[:10]
special_gamma = []
for k in range(1, len(special_idxs)+1):
    idxs = tuple(special_idxs[:k])
    g = gamma(idxs)
    plt.plot(k, g, 'rs', markersize=7)  # 红色方块标记
    special_gamma.append(g)
plt.plot(range(1, len(special_gamma)+1), special_gamma, 'r-', lw=2, label='Top-n subset γ')

# 横线标注
plt.axhline(10, ls="--", c="r", lw=1, label='γ=10')
plt.axhline(15, ls="--", c="r", lw=1, label='γ=15')

plt.xlabel("Subset size (k)")
plt.ylabel("Collinearity index γ")
plt.title("Collinearity indices for all parameter subsets")
plt.legend()
plt.tight_layout()
plt.savefig("gamma_verticals02.png", dpi=300)
plt.show()

# --------------------------------------------------------------
# 4. Step-3 —— 所有 γ 写 Excel (保持原先版本)
# --------------------------------------------------------------
rows = []
for k in range(1, len(param_names) + 1):
    for idxs in itertools.combinations(range(len(param_names)), k):
        g = gamma(idxs)
        subset_names = ", ".join(param_names[i] for i in idxs)
        rows.append({"Subset size": k, "Parameters": subset_names, "Gamma": g})

# 转 DataFrame
GammaDF = pd.DataFrame(rows)

# 按子集大小分工作表写入
with pd.ExcelWriter("gamma_subsets02.xlsx", engine="openpyxl") as writer:
    for k, grp in GammaDF.groupby("Subset size"):
        grp_sorted = grp.sort_values("Gamma", ascending=False)
        grp_sorted.to_excel(writer, sheet_name=f"size_{k}", index=False)
print("已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")

# --------------------------------------------------------------
# 5. 追加 Step-5 —— 近似标准误差 & 相关系数 (Table 8)
# --------------------------------------------------------------
print("\n=== Step-5  近似标准误差 & 相关系数 (Table-8) ===")

# —— 5.1 选“要估计的子集”：可指定共线性分析结果最优组合 ——
# 手动指定最优共线性组合（例：γ=5.22, 包含 Kbile 且敏感性排名前5）
chosen_subset_names = [
        "PRest", "Kbile", "GFR", "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab"
]
#chosen_subset_names =param_names    #待优化子集
# 自动转成下标
chosen_subset_idx = [param_names.index(p) for p in chosen_subset_names]  #记录子集下标
subset_idx = chosen_subset_idx  #子集下标
subset_names = chosen_subset_names #子集名称
# 如需恢复用Top-N灵敏度参数，只需把上面两行注释掉，改回:
# subset_idx = order[:7]
# subset_names = [param_names[i] for i in subset_idx]

p = len(subset_idx) #子集长度
print("参数子集:", subset_names)

# ============================================================
# 6. WSS 目标函数 + 最小二乘拟合
# ============================================================
print("\n=== Step‑6  WSS 目标函数 + 参数拟合优化 ===")
# ------------------------------------------------------------
#    ──把“每个人第 2 个浓度点”权重加倍
# ------------------------------------------------------------
sc_groups = []
for obs in conc_groups:                # obs 是一条实验的观测浓度数组
    sc = np.ones_like(obs)             # 先全部权重 = 1
    if len(sc) > 1:                    # 若至少有 2 个采样点
        sc[1] = 0.1                    # 让第 2 点的 sc 更小 → 权重更大
    sc_groups.append(sc)               # (权重 ∝ 1/sc)
# ------------------------------------------------------------
# 6.1 计算归一化残差
# ------------------------------------------------------------
def _residuals(theta_sub):                          # theta_sub 是待估 n 个参数的当前试探值
    full = baseline.copy()                          # 复制一份全 10 维参数向量
    full[subset_idx] = theta_sub                    # 用试探值替换 n 维敏感参数子集
    res = []                                        # 准备累计所有实验的残差
    for t, d, tinf, obs,sc in zip(                  # 同时遍历每条给药实验：
            time_groups, dose_groups,               #  ├─ t     → 采样时间点数组
            tinf_groups, conc_groups,sc_groups):    #  ├─ d,tinf→ 剂量与输注时长
        res.extend((FIT_model(t, d, tinf, *full)    #     预测浓度曲线
                   - obs)/sc)                       #   − 实测浓度 → 残差
    return np.asarray(res)                          # 返回 1-D 残差向量 (拼接所有实验)

lb = np.zeros(len(subset_idx))        # 各参数下界 0（不可负）
ub = np.full(len(subset_idx), np.inf) # 上界默认为 +∞
ub[subset_names.index("Free")] = 1.0  # 生理约束 Free≤1

# ------------------------------------------------------------
# 6.2计算归一化残差
# ------------------------------------------------------------
#----------在 SciPy 的 least_squares 框架里，“平方 + 求和”这一动作是由求解器自己完成的

opt = least_squares(                  #   Levenberg-Marquardt/Trust-Region 非线性最小二乘拟合函数
    _residuals,                       #   目标函数：加权残差
    baseline[subset_idx],             #   初值：基线生理参数
    bounds=(lb, ub),                  #   简单界限约束
    method="trf",                     #   使用 trust-region reflective 算法
    xtol=1e-10,                       #   参数步长 (step norm) 阈值
    verbose=2,                        #   输出迭代信息
    max_nfev=300)                     #   最多 300 次函数评估
theta_hat = opt.x                     #   最终最小二乘估计值，x就是使拟合误差最小的参数值。
rss = np.sum(opt.fun**2)              #   残差平方和 (cost×2)，这是文章中公式（1）计算出来的值
dof = len(opt.fun) - len(theta_hat)   #   自由度 = 数据点数 − 参数数
print(f"优化完成  RSS = {rss:.4g}  (DOF = {dof})")
print(f"success : {opt.success}")        # True / False
print(f"status  : {opt.status}")         # 0–5 的代码，见 SciPy 文档
print(f"message : {opt.message}")        # 人可读的收敛说明
print(f"nfev    : {opt.nfev}")           # 目标函数评估次数
baseline[subset_idx] = theta_hat  # 回写以便继续迭代
print(f"优化参数    : {baseline}") 

with open('saved_result/optimized_params_{today_date}.pkl', 'wb') as f:
    pickle.dump(baseline, f)
# ============================================================
# 7. 协方差 & 不确定度传播 / 验证 公式（10）
# ============================================================
print("\n=== Step‑7  协方差 + 验证 / 预测 ===")

J = opt.jac                      # ① 最终迭代点的雅可比矩阵 (∂r/∂θ),	SciPy least_squares 保存的雅可比矩阵 
sigma2 = rss / dof               # ② 残差方差估计 σ² = RSS / 自由度
cov = sigma2 * la.inv(J.T @ J)   # ③ 线性近似协方差：σ²·(JᵀJ)⁻¹
se  = np.sqrt(np.diag(cov))      # ④ 每个参数的标准误差 = 协方差对角元素开方

fit_tbl = pd.DataFrame({         # ⑤ 汇总成表便于查看 & 导出
    "Parameter": subset_names,
    "Estimate":  theta_hat,
    "StdErr":    se,
    "RelSE_%":   100 * se / np.abs(theta_hat),
})
# --------- 仍然保存参数估计表 ---------
fit_tbl.to_csv("fit_results.csv", index=False)
print("\n===== Parameter Estimates & SE =====")
print(fit_tbl)






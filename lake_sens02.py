# -*- coding: utf-8 -*-
"""
lake_sens.py —— MTX‑PBPK 模型的敏感性‑共线性分析（胡波/Brun 三步法）
-------------------------------------------------------------------
运行环境：numpy ≥ 1.20, scipy ≥ 1.6, matplotlib ≥ 3.3, pandas ≥ 1.3, openpyxl
位置：与 init_param.py 和 init_data_point4.py 同目录
输出：
  · sens_ranking.png —— 全局灵敏度条形图
  · gamma_ranges.png —— 子集大小 vs γ 区间图
  · gamma_subsets.xlsx —— 所有子集 γ 指数（按子集大小分 Sheet）
"""

import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

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
)

# 待分析参数
param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free",  "Vmax_baso", "Km_baso", "Kurine", "Kreab",
]
baseline = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
], dtype=float)

# --------------------------------------------------------------
# 1. PBPK 方程 & 模拟函数（保持与 modfit0428 一致）
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
    """返回血浆浓度‑时间曲线 (mg/L)"""
    R = dose / tinf
    y0 = np.zeros(7)
    sol = odeint(
        derivshiv,
        y0,
        t,
        args=(params, R, tinf),
        rtol=1e-6,
        atol=1e-9,
        h0=0.1,
    )
    return sol[:, 0] / VPlas
# --------------------------------------------------------------
# 2. Step‑1 —— 全局灵敏度 (RMS) 排序
# --------------------------------------------------------------
delta_rel = 0.01  # 相对扰动 1 %
sl_vectors = []    # 每个参数 sl(t) 向量
S_global   = []    # RMS 敏感度指标
# 计算基线输出，拼接成长向量
#np.concatenate([...])：把这些曲线拼接成一个长向量（表示“全部实验组的标准输出”）。
#zip(...)：将它们配成一组组 (t, d, tinf)；

# 新：构造 base_curve 及权重向量
base_curve = []
weights = []

for t, d, tinf in zip(time_groups, dose_groups, tinf_groups):
    y = FIT_model(t, d, tinf, *baseline)
    base_curve.extend(y)

    w = np.ones_like(y)
    if len(y) >= 2:
        w[1] = 2.0  # 第2个时间点权重设为2倍
    weights.extend(w)

base_curve = np.array(base_curve)
weights = np.array(weights)
# 计算加权标准差 sy
sy = np.sqrt(np.average((base_curve - np.average(base_curve, weights=weights))**2, weights=weights))

for idx, (pname, theta0) in enumerate(zip(param_names, baseline)):
    dtheta = theta0 * delta_rel if theta0 != 0 else 1e-6
    up   = baseline.copy(); up[idx] += dtheta
    down = baseline.copy(); down[idx] -= dtheta

    diff = np.concatenate([
        FIT_model(t, d, tinf, *up) - FIT_model(t, d, tinf, *down)
        for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
    ]) / (2 * dtheta)

    sl = (dtheta / sy) * (diff * weights)
    sl_vectors.append(sl)
    S_global.append(math.sqrt(np.mean(sl**2)))
# 排序并打印    
order = np.argsort(S_global)[::-1]
print("\n=== Step-1  全局灵敏度排序 (高→低) ===")
for rk, idx in enumerate(order, 1):
    print(f"{rk:2}. {param_names[idx]:<12}  S = {S_global[idx]:.3e}")
# 结果可视化
order = np.argsort(S_global)[::-1]
plt.figure(figsize=(8, 4))
plt.bar([param_names[i] for i in order], [S_global[i] for i in order])
plt.ylabel("Global sensitivity $S_l$")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("sens_ranking.png", dpi=300)

# --------------------------------------------------------------
# 3. Step‑2 —— 共线性指数 γ 计算 & γ‑包络图
# --------------------------------------------------------------

unit_vecs = [v / np.linalg.norm(v) for v in sl_vectors]

def gamma(idxs):
    """返回参数子集 idxs 的共线性指数 γ"""
    S = np.column_stack([unit_vecs[i] for i in idxs])
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)
    return 1 / math.sqrt(lam_min)

# 计算包络 (min / max) 并绘图
size_list, g_low, g_high = [], [], []
for k in range(1, len(param_names) + 1):
    g_vals = [gamma(c) for c in itertools.combinations(range(len(param_names)), k)]
    size_list.append(k)
    g_low.append(min(g_vals)); g_high.append(max(g_vals))

plt.figure()
plt.fill_between(size_list, g_low, g_high, alpha=0.3)
plt.plot(size_list, g_high, "o-", lw=1)
plt.axhline(10, ls="--", c="r"); plt.axhline(15, ls="--", c="r")
plt.xlabel("subset size"); plt.ylabel("collinearity index γ")
plt.tight_layout(); plt.savefig("gamma_ranges.png", dpi=300)



# --------------------------------------------------------------
# 4. Step‑3 —— 输出所有子集 γ 到 Excel
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
with pd.ExcelWriter("gamma_subsets.xlsx", engine="openpyxl") as writer:
    for k, grp in GammaDF.groupby("Subset size"):
        grp_sorted = grp.sort_values("Gamma", ascending=False)
        grp_sorted.to_excel(writer, sheet_name=f"size_{k}", index=False)
print("已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")

# --------------------------------------------------------------
# 5. Step‑4 —— 列出规模 2 & 3 的最坏子集（终端打印）
# --------------------------------------------------------------

def worst(k, top=5):
    combs = itertools.combinations(range(len(param_names)), k)
    ranked = sorted(((gamma(c), c) for c in combs), key=lambda x: -x[0])
    return [(g, [param_names[i] for i in idxs]) for g, idxs in ranked[:top]]

for k in (2, 3):
    print(f"\n=== γ 最高的 {k}-元子集 (前5) ===")
    for g, subset in worst(k):
        print(f"γ = {g:6.2f}  ->  {subset}")

print("\n分析完成：sens_ranking.png、gamma_ranges.png 与 gamma_subsets.xlsx 已生成。")

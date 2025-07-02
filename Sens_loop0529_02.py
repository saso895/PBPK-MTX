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
baseline_init = np.array([
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
# γ 计算工具：给 unit_vecs 与子集索引 → γ
# --------------------------------------------------------------
def gamma(idxs,unit_vecs=unit_vecs,):
    """
    unit_vecs : list[np.ndarray]  各参数单位灵敏度向量
    idxs      : tuple/list[int]   想计算的参数子集索引
    return    : γ 值 (float)
    """
    S = np.column_stack([unit_vecs[i] for i in idxs])
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)
    return 1.0 / np.sqrt(lam_min)

# --------------------------------------------------------------
# ⚙️ 把 Step-1 ~ Step-3 打包成一个函数
# --------------------------------------------------------------
def compute_sensitivity(baseline,
                        param_names,
                        time_groups, dose_groups, tinf_groups,
                        conc_groups,
                        delta_rel=0.028,
                        gamma_thresh=10.0):
    """
    返回：
    • order        —— 灵敏度从高到低的索引列表
    • S_global     —— 每个参数的 S 值列表
    • sl_vectors   —— 单参数灵敏度向量 list(np.ndarray)
    • GammaDF      —— 所有子集 γ 结果的 DataFrame
    """
    # ---------- Step-1 : 全局灵敏度 ----------
    sl_vectors, S_global = [], []

    # 先算一次 baseline 预测 & sy
    base_curve = []
    for t, d, tinf in zip(time_groups, dose_groups, tinf_groups):
        base_curve.extend(FIT_model(t, d, tinf, *baseline))
    y_obs  = np.concatenate(conc_groups)
    y_pred = np.array(base_curve)
    sy     = np.sqrt(np.mean((y_obs - y_pred) ** 2))

    # 对每个参数做 ±Δ%
    for idx, theta0 in enumerate(baseline):
        dtheta      = theta0 * delta_rel if theta0 != 0 else 1e-6
        up, down    = baseline.copy(), baseline.copy()
        up[idx]    += dtheta
        down[idx]  -= dtheta
        diff = np.concatenate([
            FIT_model(t, d, tinf, *up) - FIT_model(t, d, tinf, *down)
            for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
        ]) / (2 * dtheta)
        sl  = (dtheta / sy) * diff
        sl_vectors.append(sl)
        S_global.append(np.sqrt(np.mean(sl ** 2)))

    order = np.argsort(S_global)[::-1]

    # ---------- Step-2 / Step-3 : 算 γ & 生成 DataFrame ----------
    unit_vecs = [v / np.linalg.norm(v) for v in sl_vectors]


    rows = []
    for k in range(1, len(param_names) + 1):
        for idxs in itertools.combinations(range(len(param_names)), k):
            rows.append({
                "Subset size": k,
                "Parameters" : ", ".join(param_names[i] for i in idxs),
                "Gamma"      : gamma(idxs)
            })
            TODO:gamma函数的传递参数少了一个
    GammaDF = pd.DataFrame(rows)

    return order, S_global, sl_vectors, GammaDF


# =======================================================================
# 🔄 外层循环：重复 Step-1 ~ Step-6 直到子集稳定 & γ_max ≤ 阈值
# =======================================================================
gamma_thresh = 10.0          # 共线性收敛阈值
max_outer    = 8             # 最多循环次数
last_subset  = None          # 记录前一轮子集
baseline = baseline_init.copy()  

# --------------------------------------------------------------
# 🔁 外层循环
# --------------------------------------------------------------
for outer in range(max_outer):
    print(f"\n========  外层循环 {outer+1}/{max_outer}  ========")

    # === 计算灵敏度 + γ DataFrame ===
    (order, S_global, sl_vectors,
     GammaDF) = compute_sensitivity(
        baseline,
        param_names,
        time_groups, dose_groups, tinf_groups,
        conc_groups,
        delta_rel=0.028,
        gamma_thresh=gamma_thresh
    )

    # --- 打印灵敏度 Top-10 (保持原格式) ---
    print("\n=== Step-1  全局灵敏度排序 ===")
    for rk, idx in enumerate(order, 1):
        print(f"{rk:2}. {param_names[idx]:<12}  S = {S_global[idx]:.3e}")

    # === 下面直接进入原来的 Step-3 写 Excel ===
    GammaDF.to_excel("gamma_subsets0529.xlsx", index=False)
    print("✅已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")
    print("\n=============================================================================")
    # --------------------------------------------------------------
    # 5. 追加 Step-5 —— 近似标准误差 & 相关系数 (Table 8)
    # --------------------------------------------------------------
    print("\n=== Step-5  近似标准误差 & 相关系数 (Table-8) ===")

    # —— 5.1 选“要估计的子集”：可指定共线性分析结果最优组合 ——
    # ---------- 自动选 “γ<阈值 且 k 最大” 的参数子集 ----------
    #gamma_thresh = 10.0                           # 阈值，你也可以传入循环外的同名变量

    # Step 1: 对每个 k，判断该大小下的所有子集的 γ_max 是否 <= 阈值
    valid_k = []                                   #记录子集个数
    for k, grp in GammaDF.groupby("Subset size"):  #k是子集大小，grp是对应大小所有子集的组合
        if grp["Gamma"].max() <= gamma_thresh:     #所有组合的最大gamma值，小于10，就记录k
            valid_k.append(k)                      #得到候选的一组k                      

    # Step 2: 若有有效的 k 值，选最大的 k
    if valid_k:
        k_max = max(valid_k)                                  #选候选的一组k里最大的 k
        cands_k = GammaDF[(GammaDF["Subset size"] == k_max)]  #选出等于最大k的子集
        best_row = cands_k.loc[cands_k["Gamma"].idxmin()]     #从子集里选一组gamma最小的参数组合
        print(f"✅ 满足 γ<{gamma_thresh} 的最大参数个数为 {k_max}，选 γ 最小的子集")
    else:
        # 如果所有子集都不满足，退而求其次：选择 γ 最小的
        best_row = GammaDF.loc[GammaDF["Gamma"].idxmin()] 
        print(f"⚠️ 所有子集都存在高共线性，选 γ 最小的组合：{best_row['Parameters']}") 

    subset_names = [s.strip() for s in best_row["Parameters"].split(",")]
    subset_idx   = [param_names.index(p) for p in subset_names]
    gamma_sel = best_row["Gamma"]   

    print(f"🎯 自动选中子集 (k={len(subset_idx)}, γ={best_row['Gamma']:.2f}):", subset_names)
    print("\n=============================================================================")
    
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
    #ub[subset_names.index("Free")] = 1.0  # 生理约束 Free≤1

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
    print(f"优化次数    : {outer}") 
    df_param = pd.DataFrame({
        '参数': param_names,
        '初始参数值': baseline_init,
        '拟合参数值':  baseline
    })
    print("\n=== 参数对比 ===")
    print(df_param.to_string(index=False))
    # ---------- 用拟合后参数再算 γ ----------
    theta_tmp = baseline.copy()
    gamma_post = gamma(subset_idx)  # ← 用新的 baseline 再算 γ

    # ---------- 收敛判定：放在 Step‑6 之后 ----------
    if (subset_idx == last_subset) and (gamma_post <= gamma_thresh):
        print(f"🎯收敛判断次数    : {outer+1}")
        print(f"✅ 子集稳定，且 γ = {gamma_post:.2f} ≤ 阈值 → 收敛")
        break
    last_subset = subset_idx.copy() 
else:
    print("⚠️ 达到最大循环次数仍未满足收敛条件")

os.makedirs('saved_result', exist_ok=True)
with open('saved_result/baseline_stable.pkl', 'wb') as f:
    pickle.dump({
        'subset_idx'  : subset_idx,
        'subset_names': subset_names,
        'baseline'    : baseline,
        'gamma_max'   : gamma_sel,
    }, f)
print("🌟 稳定子集 + 参数已保存 --> saved_result/baseline_stable.pkl")

df_param = pd.DataFrame({
    '参数': param_names,
    '初始参数值': baseline_init,
    '拟合参数值':  baseline
})

print("\n=== 🏆最终优化参数对比🏆 ===")
print(df_param.to_string(index=False))
print("\n=== 🌈over🌈 ===")

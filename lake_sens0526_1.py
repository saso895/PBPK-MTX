"""
lake_sens_iter.py  —— 迭代执行 Step‑3 ~ Step‑6，直到获得
                   稳定且共线性可接受的可估计参数子集。
================================================================
使用你的 **lake_sens0526.py** 相同的变量命名与数据接口：
    • param_names        —— 10 个全局待选参数名称
    • baseline           —— 对应的 10 维基线参数向量
    • FIT_model()        —— 计算血浆浓度曲线（保持签名不变）
    • derivshiv()        —— PBPK 微分方程（从原脚本复用）
    • time_groups, dose_groups, tinf_groups, conc_groups —— 训练数据
因此，你可以直接把本文件放到与原脚本同级目录运行：
    python lake_sens_iter.py
----------------------------------------------------------------
迭代逻辑
========
1. **全局灵敏度 (Step‑3)**
   - 相对扰动 δ=3%（可调）
   - 用中央差分近似偏导，计算公式 (4‑6) 得到 S_global。
2. **共线性分析 (Step‑4)**
   - 按 S_global 从高到低尝试加入参数，若 γ(子集) ≤ γ_max(默认10) 就保留。
3. **最小二乘拟合 (Step‑6)**
   - 仅拟合当前子集，用 SciPy `least_squares`；
   - 计算残差平方和 → 标准误差 SE → 相对误差 RelSE%。
4. **收敛判定**
   - 子集两轮不变，且 **所有** 拟合参数相对变化 < tol_param(1e‑3)。
----------------------------------------------------------------
输出
====
• `fit_results_iter.csv`  —— 最终子集 θ̂/SE/RelSE%
• `optimized_params_iter_YYYY‑MM‑DD.pkl` —— 收敛后的完整 10 维参数
----------------------------------------------------------------
"""
from __future__ import annotations
import math, datetime, pickle, itertools
import numpy as np
import numpy.linalg as la
from scipy.integrate import odeint
from scipy.optimize import least_squares

# ------------------------------------------------------------------
# 0. 载入固定参数 & 观测数据（保持与原 lake_sens0526.py 相同接口）
# ------------------------------------------------------------------
from init_param import (
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
    PRest, PK, PL, Kbile, GFR, Free,
    Vmax_baso, Km_baso, Kurine, Kreab,
)
from init_data_point4 import (
    time_points_train as time_groups,
    input_dose_train  as dose_groups,
    inject_timelen_train as tinf_groups,
    concentration_data_train as conc_groups,
)

param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab",
]
baseline = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free, Vmax_baso, Km_baso, Kurine, Kreab,
], dtype=float)

# ------------------------------------------------------------------
# 1. PBPK 方程 —— 直接复用你原脚本的 derivshiv & FIT_model
# ------------------------------------------------------------------

def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    inp = R if t <= T_total else 0.0
    dy = np.zeros(7)
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
        rtol=1e-6, atol=1e-9,
    )
    return sol[:, 0] / VPlas  # 返回血浆浓度

# ------------------------------------------------------------------
# 2. 敏感度 (Step‑3) & 共线性 (Step‑4)
# ------------------------------------------------------------------

def calc_sensitivity(baseline: np.ndarray, delta_rel: float = 0.03):
    """返回 (sl_vectors, S_global) 列表"""
    # 基线预测 & 标度 sy
    base_curve = np.concatenate([
        FIT_model(t, d, tinf, *baseline)
        for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
    ])
    y_obs = np.concatenate(conc_groups)
    sy = np.sqrt(np.mean((y_obs - base_curve) ** 2))

    sl_vecs, Sg = [], []
    for idx, theta0 in enumerate(baseline):
        dtheta = theta0 * delta_rel if theta0 != 0 else 1e-6
        up, down = baseline.copy(), baseline.copy()
        up[idx] += dtheta; down[idx] -= dtheta
        diff = np.concatenate([
            FIT_model(t, d, tinf, *up) - FIT_model(t, d, tinf, *down)
            for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
        ]) / (2 * dtheta)
        sl = (dtheta / sy) * diff
        sl_vecs.append(sl)
        Sg.append(math.sqrt(np.mean(sl ** 2)))
    return sl_vecs, Sg

def gamma(unit_vecs, idxs):
    """根据公式 (8) 计算子集 γ"""
    if len(idxs) <= 1:
        return 1.0
    S = np.column_stack([unit_vecs[i] for i in idxs])
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)
    return 1.0 / math.sqrt(lam_min)

# ------------------------------------------------------------------
# 3. 迭代主程序
# ------------------------------------------------------------------

def iterate(max_iter: int = 10, gamma_max: float = 10.0, tol_param: float = 1e-3):
    base = baseline.copy()
    subset_prev, theta_prev = None, None

    for it in range(1, max_iter + 1):
        print(f"\n===== ITER {it} =====")
        # Step‑3
        sl_vecs, S_global = calc_sensitivity(base)
        order = np.argsort(S_global)[::-1]
        unit_vecs = [v / la.norm(v) for v in sl_vecs]

        # Step‑4  —— 逐个加入高灵敏度参数，γ 不超阈值就保留
        subset_idx = []
        for idx in order:
            test = subset_idx + [idx]
            if gamma(unit_vecs, test) <= gamma_max:
                subset_idx = test
        subset_names = [param_names[i] for i in subset_idx]
        g_val = gamma(unit_vecs, subset_idx)
        print("子集:", subset_names, "  γ =", round(g_val, 2))

        # Step‑6  —— 最小二乘
        def _res(theta_sub):
            full = base.copy(); full[subset_idx] = theta_sub
            res = []
            for t, d, tinf, obs in zip(time_groups, dose_groups, tinf_groups, conc_groups):
                res.extend(FIT_model(t, d, tinf, *full) - obs)
            return np.asarray(res)

        theta0 = base[subset_idx]
        opt = least_squares(_res, theta0, method="trf", xtol=1e-10)
        base[subset_idx] = opt.x
        rss, dof = np.sum(opt.fun ** 2), len(opt.fun) - len(opt.x)
        se = np.sqrt(np.diag((rss / dof) * la.inv(opt.jac.T @ opt.jac)))
        relse = 100 * se / np.abs(opt.x)
        print("RelSE%:", np.round(relse, 1))

        # 收敛判定
        subset_same = (subset_names == subset_prev)
        theta_small = False
        if theta_prev is not None and theta_prev.size == opt.x.size:
            theta_small = np.all(np.abs((opt.x - theta_prev) / theta_prev) < tol_param)
        subset_prev, theta_prev = subset_names, opt.x.copy()
        if subset_same and theta_small:
            print("→ 收敛：子集稳定且参数变化 < tol_param")
            break

    # 保存结果
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    fit_tbl = {
        "Parameter": subset_names,
        "Estimate": opt.x,
        "StdErr": se,
        "RelSE_%": relse,
    }
    import pandas as pd
    pd.DataFrame(fit_tbl).to_csv("fit_results_iter.csv", index=False)
    with open(f"optimized_params_iter_{today}.pkl", "wb") as f:
        pickle.dump(base, f)
    print("\n最终子集与参数已保存。文件: fit_results_iter.csv / optimized_params_iter_*.pkl")


if __name__ == "__main__":
    iterate()

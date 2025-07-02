# -*- coding: utf-8 -*-
"""
Fully‑refactored PBPK‑MCMC fitting script based on *modmcmc0507.py*.
===============================================================
核心改动 ⭐⭐⭐ (对应之前给出的全部建议)
--------------------------------------------------------------
1. **Aesara‑compatible ODE**  ➜  使用 `pm.ode.DifferentialEquation` 让梯度可追踪，从而可用 NUTS。
2. **单一观测节点**          ➜  把 51 位受试者的观测拼接成一个向量，极大精简计算图。
3. **明确的 MAP 起点**        ➜  先 `pm.find_MAP()`，随后 NUTS；必要时可切换 `pm.Metropolis()`。
4. **函数式拆分**              ➜  数据读取 / ODE / 建模 / 采样 各自独立，方便测试与替换。
5. **可选黑盒 SMC**           ➜  若梯度失效可一键改 `sampler="SMC"`。
6. **tqdm 监控 & CLI 参数**    ➜  运行时进度可视化，Python `argparse` 自定义采样轮数。

使用方式
---------
```bash
python modmcmc0507_refactor.py  \
       --draws 2000 --tune 1000 --chains 4 \
       --sampler NUTS            # 或 "SMC" / "Metropolis"
```
生成的样本会保存在 `trace_modmcmc0507.netcdf`。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at
from pymc.ode import DifferentialEquation
import arviz as az
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 0 | 常量和辅助工具
# -----------------------------------------------------------------------------

# >>> 读取 PBPK 全局解剖学常量（示例，按需替换） >>>
QRest, QK, QL, QPlas = 4.2, 1.2, 1.8, 5.0  # L/h
VRest, VK, VL, VPlas = 42., 1.8, 1.5, 3.0  # L
# <<<--------------------------------------------------------------------------

def load_data():
    """示例数据加载函数——请替换为你自己的 CSV / pickle 读取逻辑。"""
    # 假设 data.csv 里包含列: subject, time, dose, conc
    df = pd.read_csv("data.csv")
    subjects = df["subject"].unique()

    # 拼成向量
    conc_obs = df["conc"].values.astype("float64")
    times = df["time"].values.astype("float64")
    subj_idx = df["subject"].astype("int64").values  # 0…N‑1
    return times, conc_obs, subj_idx, subjects

# -----------------------------------------------------------------------------
# 1 | 可微分 PBPK ODE
# -----------------------------------------------------------------------------

STATE_NAMES = [
    "Plasma",  # y[0]
    "Liver",
    "Kidney",
    "Rest",
    "Urine",
]

PARAM_NAMES = [
    "PRest",
    "PK",
    "PL",
    "Kbile",
    "GFR",
    "Free",
    "Vmax_baso",
    "Km_baso",
    "Kurine",
    "Kreab",
]


def pbpk_deriv(y, t, theta):
    """Return dy/dt for the PBPK system.

    Parameters
    ----------
    y : tensor (n_state,)
    t : scalar (ignored here, non‑autonomous terms可手动加)
    theta : tensor (10,)
    """
    (
        PRest,
        PK,
        PL,
        Kbile,
        GFR,
        Free,
        Vmax_baso,
        Km_baso,
        Kurine,
        Kreab,
    ) = theta

    yPlas, yLiv, yKid, yRest, yUrine = y  # unpack for clarity

    # >>>> 药物输入（示例：单次静脉注射 100 mg at t=0） >>>>
    D_total = 100.0  # mg
    input_rate = at.switch(at.le(t, 0.01), D_total / 0.01, 0.0)  # delta‑like bolus
    # <<<<----------------------------------------------------------------------

    dydt = at.stack([
        (QRest * yRest / VRest / PRest)
        + (QK * yKid / VK / PK)
        + (QL * yLiv / VL / PL)
        - (QPlas * yPlas / VPlas)
        + Kreab * yUrine
        + input_rate / VPlas,
        (QPlas * yPlas / VPlas)
        - (QL * yLiv / VL / PL)
        - Kbile * Free * yLiv / VL,
        (QPlas * yPlas / VPlas)
        - (QK * yKid / VK / PK)
        - GFR * Free * yKid / VK,
        (QPlas * yPlas / VPlas)
        - (QRest * yRest / VRest / PRest),
        GFR * Free * yKid / VK - Kreab * yUrine - Kurine * yUrine,
    ])
    return dydt

# DifferentialEquation wrapper (Aesara‑compatible)
ODE_SYSTEM = DifferentialEquation(
    func=pbpk_deriv,
    times=np.linspace(0, 24, 121),  # 0–24 h, 每 0.2 h 一点; 会被实际观测 time 替换
    n_states=len(STATE_NAMES),
    n_theta=len(PARAM_NAMES),
    t0=0.0,
)

# -----------------------------------------------------------------------------
# 2 | 构建并采样模型
# -----------------------------------------------------------------------------

def build_and_sample(times, conc_obs, subj_idx, sampler="NUTS", draws=2000, tune=1000, chains=4):
    """核心入口：返回 `az.InferenceData`。"""

    unique_subj = np.unique(subj_idx)
    n_subjects = len(unique_subj)

    # 用 broadcast trick 一次性算 N×T 轨迹：
    times_shared = np.tile(times[:, None], (1, n_subjects)).T  # shape (N, T)

    with pm.Model() as model:
        # ========== Priors ==========
        prior_mu = np.log([1, 1, 1, 0.1, 0.1, 0.5, 10, 1, 0.1, 0.1])
        prior_sigma = 0.5
        theta = pm.LogNormal("theta", mu=prior_mu, sigma=prior_sigma, shape=len(PARAM_NAMES))

        # 初始条件统一设 0
        y0 = pm.MutableData("y0", np.zeros(len(STATE_NAMES)))

        # ========== Solve ODE ==========
        y_hat_full = ODE_SYSTEM(y0=y0, theta=theta, times=times)
        CA_plasma_full = y_hat_full[:, 0] / VPlas  # shape (T,)

        # 如果每个受试者有不同剂量，可在 theta 增加层级或对 y0 做广播
        # 这里假设同一剂量 ➜ 直接 tile
        mu_vec = at.repeat(CA_plasma_full, n_subjects)  # (T×N,)

        # ========== Likelihood ==========
        sigma = pm.HalfNormal("sigma", 5.0)
        y_obs = pm.Normal("y_obs", mu=mu_vec, sigma=sigma, observed=conc_obs)

        # ========== Sampling strategy ==========
        init_map = pm.find_MAP(progressbar=True)

        if sampler.upper() == "NUTS":
            step = pm.NUTS()
        elif sampler.upper() == "METROPOLIS":
            step = pm.Metropolis()
        elif sampler.upper() == "SMC":
            trace = pm.sample_smc(draws=draws)
            return trace
        else:
            raise ValueError("Unknown sampler: " + sampler)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            start=init_map,
            step=step,
            chains=chains,
            target_accept=0.9,
            progressbar=True,
            cores=chains,
        )

    return trace

# -----------------------------------------------------------------------------
# 3 | CLI 入口
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PBPK‑MCMC parameter estimation")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--sampler", choices=["NUTS", "Metropolis", "SMC"], default="NUTS")
    args = parser.parse_args()

    times, conc_obs, subj_idx, _ = load_data()

    trace = build_and_sample(
        times=times,
        conc_obs=conc_obs,
        subj_idx=subj_idx,
        sampler=args.sampler,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
    )

    # 保存 netcdf 方便后处理
    out_path = Path("trace_modmcmc0507.nc")
    az.to_netcdf(trace, out_path)
    print(f"Trace saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()

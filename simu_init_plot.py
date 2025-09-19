# -*- coding: utf-8 -*-
"""
Simu.py  ‑‑ 使用 **初始参数** 对所有受试者进行 MTX 浓度预测，
          绘制 95% 浓度置信区间、GOF 图（含 5–95% 预测带 + $R^2$），并计算 AFE/AAFE。

保持原有变量名与总体逻辑，但统一重构为函数 + main，
方便脚本/单元测试两用。

2025‑07‑29  更新内容
---------------------------------------------
1. 每个受试者的浓度‑时间曲线绘制在**独立子图**中，并将所有子图排布在同一画布（可自动布局）。
2. 在 GOF 图左上角加入 $R^2$ 注释。
"""
import datetime
import math
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== 业务模块 (保持原始引用) =========================================
from init_data_point4 import (
    time_points_train,
    concentration_data_train,
    input_dose_train,
    inject_timelen_train,
)
from init_param0723 import init_pars  # 14‑dim dict
from ode_core0723 import PK_model     # 细网格模拟函数

from init_param0723 import (
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
    # === MOD BEGIN 2025‑07‑23 新增参数引入 ===
    Vmax_apical, Km_apical, Vmax_bile, Km_bile
    # === MOD END ============================
)

# ==== 常量 & 路径 ======================================================
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
SAVE_DIR = Path("saved_result")
SAVE_DIR.mkdir(exist_ok=True)

baseline_init = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free, Vmax_baso, Km_baso, Kurine, Kreab,
    # === MOD BEGIN 2025‑07‑23 新增参数引入 ===
    Vmax_apical, Km_apical, Vmax_bile, Km_bile
    # === MOD END ============================
], dtype=float)

# ----------------------------------------------------------------------
# 1. 加载参数（保持原接口）
# ----------------------------------------------------------------------

def load_parameters(source: str = "init", file_path: Optional[str] = None, idx: Optional[int] = None) -> List[float]:
    """灵活加载 14 维参数向量；默认返回 *初始参数*。"""
    if source == "init":
        return baseline_init  # ← 14‑dim

    if file_path is None:
        raise ValueError(f"source={source} 需要提供 file_path")

    with open(file_path, "rb") as f:
        loaded = pickle.load(f)

    # modfit: ndarray 或 dict{baseline: ...}
    if source == "modfit":
        if isinstance(loaded, dict):
            loaded = loaded.get("baseline", loaded.get("params"))
        return np.asarray(loaded)[:14].tolist()

    # mcmc: 多链 dict 或 ndarray
    if source == "mcmc":
        if isinstance(loaded, dict):
            if idx is None:
                idx = 1
            key = f"chain{idx}_params" if f"chain{idx}_params" in loaded else list(loaded.keys())[0]
            loaded = loaded[key]
        return np.asarray(loaded)[:14].tolist()

    # 任意 pkl
    if source == "file":
        return np.asarray(loaded)[:14].tolist()

    raise ValueError(f"未知 source: {source}")

# ----------------------------------------------------------------------
# 2. 统计指标
# ----------------------------------------------------------------------

def calc_afe_aafe(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """返回 (AFE, AAFE)，两者均以 10 为底。"""
    log_ratio = np.log10(pred / obs)
    afe = 10 ** (log_ratio.mean())
    aafe = 10 ** (np.abs(log_ratio).mean())
    return afe, aafe


def calc_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """计算决定系数 R^2（线性尺度）。"""
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------

def main():
    # 3.1 选择参数
    params = load_parameters("init")

    # 3.2 容器
    y_mu: List[np.ndarray] = []        # 细网格预测 (Time, Conc)
    obs_all: List[float] = []
    pred_all: List[float] = []
    afe_list: List[Tuple[int, float, float]] = []  # (case, AFE, AAFE)

    # 3.3 逐病例模拟
    iterator = tqdm(range(len(time_points_train)), desc="Simulating")
    for i in iterator:
        tp_obs   = np.asarray(time_points_train[i], dtype=float)
        conc_obs = np.asarray(concentration_data_train[i], dtype=float) + 1e-5  # 防除零
        dose     = float(input_dose_train[i])
        tinf     = float(inject_timelen_train[i])
        duration = tp_obs[-1]

        # ---- 预测浓度 (细网格 0.1 h)
        pred_profile = PK_model(tp_obs, dose, tinf, duration, *params)  # (N,2)
        y_mu.append(pred_profile)

        # ---- 匹配观测时间点（线性插值）
        pred_at_obs = np.interp(tp_obs, pred_profile[:, 0], pred_profile[:, 1])
        obs_all.extend(conc_obs.tolist())
        pred_all.extend(pred_at_obs.tolist())

        # ---- 单病例 AFE / AAFE
        afe_i, aafe_i = calc_afe_aafe(conc_obs, pred_at_obs)
        afe_list.append((i + 1, afe_i, aafe_i))

    # 3.4 全局统计 & CI 计算
    obs_arr  = np.asarray(obs_all)
    pred_arr = np.asarray(pred_all)
    afe_all, aafe_all = calc_afe_aafe(obs_arr, pred_arr)
    r2_all = calc_r2(obs_arr, pred_arr)


    # ---------- 置信区间倍率（确保下界 < 上界） -----------------------
    log_res = np.log10(pred_arr / obs_arr)
    ci_low,  ci_up  = sorted(10 ** np.quantile(log_res, [0.025, 0.975]))
    iqr_low, iqr_up = sorted(10 ** np.quantile(log_res, [0.25, 0.75]))
    band_low, band_up = sorted(10 ** np.quantile(log_res, [0.05, 0.95]))

    # ------------------------------------------------------------------
    # 4. 绘图：每个病例单独子图 + 95% 置信区
    # ------------------------------------------------------------------
    n_cases = len(y_mu)
    ncols = 4                 # 每行子图数量，可根据需要调整
    nrows = int(math.ceil(n_cases / ncols))
    fig_ct, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=False, sharey=False)
    axes = axes.flatten()

    for k in range(n_cases):
        ax = axes[k]
        tp   = y_mu[k][:, 0]
        cp = y_mu[k][:, 1]
        # 95 % CI
        ax.fill_between(tp, cp * ci_low, cp * ci_up, color="#6fa8dc", alpha=0.25, label="95 % CI" if k == 0 else "")
        # IQR
        ax.fill_between(tp, cp * iqr_low, cp * iqr_up, color="#0b5394", alpha=0.35, label="IQR" if k == 0 else "")
        # Median prediction

        # 预测曲线 + 95% CI
        ax.plot(tp, cp, color="k", lw=1)

        # 观测点
        ax.scatter(time_points_train[k], concentration_data_train[k], facecolors="none", edgecolors="red", s=20)        
        ax.set_title(f"Case {k + 1}")
        if k % ncols == 0:
            ax.set_ylabel("Concentration (mg/L)")
        if k >= (n_cases - ncols):
            ax.set_xlabel("Time (h)")

    # 删除空白子图（如有）
    for idx in range(n_cases, len(axes)):
        fig_ct.delaxes(axes[idx])

    fig_ct.tight_layout()
    conc_svg = SAVE_DIR / f"simu_init_{TODAY}.svg"
    fig_ct.savefig(conc_svg, format="svg", bbox_inches="tight")
    plt.close(fig_ct)

    # ------------------------------------------------------------------
    # 5. GOF 图（对数‑对数，含 5–95% 预测带 + R^2）
    # ------------------------------------------------------------------
    fig_gof, ax_gof = plt.subplots(figsize=(6, 6))
    ax_gof.scatter(obs_arr, pred_arr, s=15, color="#1f77b4", label="Cases")
    min_val, max_val = obs_arr.min(), obs_arr.max()

    # 5.1 预测带 (灰色填充)
    x_band = np.logspace(np.log10(min_val), np.log10(max_val), 200)
    ax_gof.fill_between(x_band, x_band * band_low, x_band * band_up,
                        color="lightgrey", alpha=0.5, label="5–95% PI")

    # 5.2 参考线
    ax_gof.plot([min_val, max_val], [min_val, max_val], "k-", label="y = x")
    ax_gof.plot([min_val, max_val], [min_val * 10, max_val * 10], "k--", lw=0.6, label="×10 / ÷10")
    ax_gof.plot([min_val, max_val], [min_val / 10, max_val / 10], "k--", lw=0.6)

    # R^2 注释 (左上)
    ax_gof.text(0.05, 0.95, fr"$R^2 = {r2_all:.3f}$", transform=ax_gof.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax_gof.set_xscale("log", base=10)
    ax_gof.set_yscale("log", base=10)
    ax_gof.set_xlabel("Observed (mg/L)")
    ax_gof.set_ylabel("Predicted (mg/L)")
    ax_gof.set_xlim(min_val, max_val)
    ax_gof.set_ylim(min_val, max_val)
    ax_gof.set_aspect("equal", "box")
    ax_gof.legend()
    gof_svg = SAVE_DIR / f"gof_init_{TODAY}.svg"
    fig_gof.savefig(gof_svg, format="svg", bbox_inches="tight")
    plt.close(fig_gof)

    # ------------------------------------------------------------------
    # 6. 保存 AFE / AAFE
    # ------------------------------------------------------------------
    df_metrics = pd.DataFrame(afe_list, columns=["Case", "AFE", "AAFE"])
    df_metrics.loc[len(df_metrics)] = ["ALL", afe_all, aafe_all]
    excel_path = SAVE_DIR / f"afe_aafe_init_{TODAY}.xlsx"
    df_metrics.to_excel(excel_path, index=False)

    # ------------------------------------------------------------------
    # 7. 控制台摘要
    # ------------------------------------------------------------------
    print("\n✓ 浓度曲线保存 →", conc_svg)
    print("✓ GOF 图保存  →", gof_svg)
    print("✓ AFE/AAFE 保存→", excel_path)
    print("    Overall  AFE  = %.3g" % afe_all)
    print("    Overall AAFE  = %.3g" % aafe_all)
    print("    Overall  R^2  = %.3g" % r2_all)


# ==== CLI =============================================================
if __name__ == "__main__":
    main()

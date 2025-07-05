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
import os,pickle,datetime
from tqdm import tqdm

# >>> MOD 0 : 新增 SALib 依赖
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
# <<< MOD 0 --------------------------------------------------

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
LOCKED = ["Km_baso"] 
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
        + inp #/ VPlas
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

# =======================================================================
# 🔄 外层循环：重复 Step-1 ~ Step-6 直到子集稳定 & γ_max ≤ 阈值
# =======================================================================
gamma_thresh = 10.0          # 共线性收敛阈值
max_outer    = 8             # 最多循环次数
last_subset  = None          # 记录前一轮子集
baseline = baseline_init.copy()  
for outer in range(max_outer):
    print(f"\n========  外层循环 {outer+1}/{max_outer}  ========")
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # >>> MOD 1 : Step-1 —— Morris 全局灵敏度 (SALib)
    # --------------------------------------------------------------
    # ---------- 封装：单次模型评估 ----------
    def model_eval(theta_vec):
        """
        theta_vec: 1-D ndarray (n_param,)
        返回拼接在一起的所有实验输出 (flatten)
        """
        out = []
        for tt, d, ti in zip(time_groups, dose_groups, tinf_groups):
            out.extend(FIT_model(tt, d, ti, *theta_vec))
        return np.asarray(out) 
    
    print("\n=== Step-1  Morris 全局灵敏度 (SALib) ===")

    # *1.1* 定义 SALib problem 字典（用 ±30% 作上下界，示例）
    bounds = np.column_stack([
        baseline * 0.7,
        baseline * 1.3
    ])

    # ====== ☆ 生 理 上 下 限 ☆ ======
    limit_dict = {
        "PRest": (0.5, 3.0),
        "PK":    (1.0, 4.0),
        "PL":    (1.0, 8.0),
        "Kbile": (0.01, 0.3),
        "GFR":   (30, 150),
        "Free":  (0.05, 0.15),
        "Vmax_baso": (0.1, 1e3),
        "Km_baso":   (1, 500),
        "Kurine":    (0.01, 10),
        "Kreab":     (0.0, 1),
    }
    # --- 覆写 bounds 中受限参数 ---
    for i, pname in enumerate(param_names):
        if pname in limit_dict:
            bounds[i, 0] = limit_dict[pname][0]        # 下限
            bounds[i, 1] = limit_dict[pname][1]        # 上限

    problem = dict(num_vars=len(param_names),
               names=param_names,
               bounds=bounds.tolist())

    # *1.2* 生成 Morris 采样
    N_TRAJ   = 40      # 轨迹条数，可视情况调
    X = morris_sample.sample(problem, N_TRAJ, num_levels=4,
                              optimal_trajectories=None)
    
    print("模型批量运行 …")
    Y_full = np.array([model_eval(row) for row in tqdm(X)])   # (1200, M)

    # ★★★ 关键改动：把多输出压缩成一维标量 ★★★
# === MOD BEGIN Step-1 指标对齐（AUC → log-RMSE） ====================
    # ① 把观测浓度展平成一维
    obs_flat = np.concatenate(conc_groups)                       # (M,)
    obs_clip  = np.clip(obs_flat, 1e-9, None)
    # ② 对每条采样的预测曲线计算 log10-RMSE
    pred_flat = Y_full.reshape(Y_full.shape[0], -1) 
    pred_clip = np.clip(pred_flat, 1e-9, None)             # (N, M)
    log_err   = np.log10(pred_clip  + 1e-9) - np.log10(obs_clip)  # 避免 log(0)
    Y_scalar  = np.sqrt(np.mean(log_err**2, axis=1))             # (N,)

# === MOD END Step-1 ===================================================

    # *1.4* Morris 分析：对每个 time-point 取均方根后再汇总
    Si = morris_analyze.analyze(problem, X, Y_scalar, 
                                     conf_level=0.95,
                                     print_to_console=False)

    mu_star = Si['mu_star']      # ① 绝对均值
    sigma    = Si['sigma']
    # 输出结果
    gsa_df = pd.DataFrame({
        'param'  : param_names,
        'mu_star': mu_star,
        'sigma'  : sigma
    })
    gsa_df.to_csv(f'saved_result/morris_result{today_date}.csv', index=False)
    print(gsa_df.sort_values('mu_star', ascending=False))

    infl_mask = mu_star >= 0.1
    param_ids = np.where(infl_mask)[0]

    # === NEW：锁定不想参与后续 γ 与拟合的参数 =====================
    #lock = ["Km_baso"]                           # 需要固定的参数名，可一次写多个
    param_ids = [i for i in param_ids
             if param_names[i] not in LOCKED]  # 过滤掉锁定项
 
    print(f"★ Morris 保留下来做 γ 分析的参数数目(锁 {LOCKED} 后): "
        f"{len(param_ids)} / {len(param_names)}")

    ### MOD 1-3 ：提供排序数组 order 供后面绘图 ###
    order = np.argsort(mu_star)[::-1]

    # ==================================================
    # Step-2  γ_max / 共线性分析
    # ==================================================
    def local_sensitivity(theta_base, eps=1e-6):
        """
        计算局部灵敏度矩阵 S  (n_out, n_param)
        Forward difference；eps 可改为 1e-20j 使用复步
        """
        y0 = model_eval(theta_base)
        n_out, n_par = y0.size, theta_base.size
        S = np.empty((n_out, n_par))
        for j in range(n_par):
            tpert = theta_base.copy()
            tpert[j] *= (1.0 + eps)
            y1 = model_eval(tpert)
            S[:, j] = (y1 - y0) / (theta_base[j] * eps)
        return S

    print("\n[Step-2] 计算局部灵敏度矩阵 & γ_max")
    theta0 = baseline.copy()            # 基线子集
    sl_vectors = local_sensitivity(theta0) # 用作后续 γ 计算
    unit_vecs  = [v/la.norm(v) for v in sl_vectors.T]      # 注意转置(.T)
    
    def gamma(idxs):
        S = np.column_stack([unit_vecs[i] for i in idxs])#公式7的小s构造出大S
        lam_min = np.min(np.linalg.eigvals(S.T @ S).real)#公式8根号里的内容
        return 1/math.sqrt(lam_min)#公式8最终结果

    # ========= γ-Envelope 图（可选） =========
    plt.figure(figsize=(6,4))
    for k in range(1, len(param_names)+1):
        g_all = [gamma(c) for c in itertools.combinations(range(len(param_names)), k)]
        plt.plot([k]*len(g_all), g_all, 'k.', alpha=.4, ms=3)
    # “Top-n” 曲线
    topn_gamma = [gamma(tuple(order[:k])) for k in range(1, len(order)+1)]
    plt.plot(range(1,11), topn_gamma[:10], 'r-s', lw=2, label='Top-n')
    plt.axhline(10, ls='--', c='r')
    plt.xlabel('Subset size k')
    plt.ylabel('γ')
    plt.tight_layout()
    save_path =f'saved_result/gamma_{today_date}_{outer+1}.png' 
    plt.savefig(save_path, dpi=300)
    #plt.show()
    # --------------------------------------------------------------
    # 4. Step-3 —— 所有 γ 写 Excel (保持原先版本)
    # --------------------------------------------------------------
    allowed = [i for i, name in enumerate(param_names) if name not in LOCKED]
    rows = []
    for k in range(1, len(allowed) + 1):
        for idxs in itertools.combinations(allowed, k):
            g = gamma(idxs)
            subset_names = ", ".join(param_names[i] for i in idxs)
            rows.append({"Subset size": k, "Parameters": subset_names, "Gamma": g})   
    print("已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")
    # 转 DataFrame
    GammaDF = pd.DataFrame(rows)
    print(f"γ_max 本循环 = {GammaDF['Gamma'].max():.1f}")
    # 按子集大小分工作表写入
    save_path =f'saved_result/gamma_subsets{today_date}.xlsx' 
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for k, grp in GammaDF.groupby("Subset size"):
            grp_sorted = grp.sort_values("Gamma", ascending=False)
            grp_sorted.to_excel(writer, sheet_name=f"size_{k}", index=False)
    print("已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")
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
        if grp["Gamma"].min() <= gamma_thresh:     #所有组合的最大gamma值，小于10，就记录k
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

    print(f"✔️自动选中子集 (k={len(subset_idx)}, γ={best_row['Gamma']:.2f}):", subset_names)

    
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
            pred = FIT_model(t, d, tinf, *full)
            # === MOD BEGIN ② clip 残差 ===================================
            pred_clip = np.clip(pred, 1e-9, None)
            obs_clip  = np.clip(obs,  1e-9, None)
            res.extend((np.log10(pred_clip) - np.log10(obs_clip)) / sc)
# === MOD END =================================================
            #res.extend((np.log10(pred + 1e-9) - np.log10(obs)) / sc)
        return np.asarray(res)                          # 返回 1-D 残差向量 (拼接所有实验)

    lb = np.zeros(len(subset_idx))        # 各参数下界 0（不可负）
    ub = np.full(len(subset_idx), np.inf) # 上界默认为 +∞
    #ub[subset_names.index("Free")] = 1.0  # 生理约束 Free≤1
    for i, pname in enumerate(subset_names):
        if pname in limit_dict:           # limit_dict 在 131–147 行已给出
            lb[i], ub[i] = limit_dict[pname]
    # ------------------------------------------------------------
    # 6.2计算归一化残差
    # ------------------------------------------------------------
    #----------在 SciPy 的 least_squares 框架里，“平方 + 求和”这一动作是由求解器自己完成的
    # 如果 baseline 中某些参数初始值超出 bounds，裁剪到合法范围
    x0 = baseline[subset_idx].copy()
    x0 = np.clip(x0, lb, ub)  # 防止 initial guess 落在边界外
    opt = least_squares(                  #   Levenberg-Marquardt/Trust-Region 非线性最小二乘拟合函数
        _residuals,                       #   目标函数：加权残差
        x0,
        #baseline[subset_idx],             #   初值：基线生理参数
        bounds=(lb, ub),                  #   简单界限约束
        method="trf",                     #   使用 trust-region reflective 算法
        jac='3-point',
        #diff_step=baseline*1e-4,
        diff_step=np.maximum(np.abs(baseline[subset_idx])*1e-4, 1e-6),#baseline[subset_idx] * 1e-4,   # ★ 改这里！
        x_scale='jac',
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
    # ----------------------------------------------------------
    same_subset = (last_subset is not None) and (set(subset_idx) == set(last_subset))
    gamma_post  = gamma(subset_idx)          # 子集 γ
    if same_subset and (gamma_post <= gamma_thresh):
        print(f"✅ 子集稳定，且 γ = {gamma_post:.2f} ≤ {gamma_thresh} → 收敛")
        break

    last_subset = subset_idx.copy()
    # --------------------------------------------------------
else:
    print("⚠️ 达到最大循环次数仍未满足收敛条件")

#os.makedirs('saved_result', exist_ok=True)
save_path =f'saved_result/SA_param{today_date}.pkl' 
with open(save_path, 'wb') as f:
    pickle.dump({
        'subset_idx'  : subset_idx,
        'subset_names': subset_names,
        'baseline'    : baseline,
        'gamma_max'   : gamma_sel,
    }, f)
print(f"🌟 稳定子集 + 参数已保存 --> {save_path} ")

# —— 估计最优点附近的协方差，用作 MCMC proposal —— 
# ——仅当子集稳定 & 优化成功后才估协方差——
if  opt.success:           # ✱① 双保险
    jac = opt.jac                           # n_data × n_param
    jt_j = jac.T @ jac

    # 若 JTJ 奇异 → 用伪逆
    try:
        cov = np.linalg.inv(jt_j)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jt_j)          # ✱② 奇异矩阵 fallback
        print("⚠️  JTJ 奇异，已用伪逆近似协方差矩阵")

    #os.makedirs('saved_result', exist_ok=True)
    save_path =f'saved_result/opt_cov{today_date}.pkl' 
    with open(save_path, 'wb') as f:
        pickle.dump(cov, f)
    print("📦 已保存协方差矩阵 opt_cov.pkl，可作为 MCMC proposal_cov")

df_param = pd.DataFrame({
    '参数': param_names,
    '初始参数值': baseline_init,
    '拟合参数值':  baseline
})

print("\n=== 🏆最终优化参数对比🏆 ===")
print(df_param.to_string(index=False))
print("\n=== 🌈over🌈 ===")

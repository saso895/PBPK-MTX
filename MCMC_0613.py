"""
自写 Metropolis-Hastings 采样器，用 0427_Powell 初步拟合结果做先验中心。
采完链后把后验均值保存为 saved_result/mcmc_params0427.pkl，
可直接被 Simu.py / simu_plot.py 调用。
运行方式:
    python mcmc_metropolis0427.py
"""

import numpy as np
from tqdm import tqdm, trange
import pickle, time, os, datetime,pandas as pd
from scipy.integrate import odeint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit                 # === MOD-1: JIT
from joblib import Parallel, delayed
#from tqdm.contrib.concurrent import tqdm_joblib      # 让 joblib 也有进度条

#from tqdm.contrib.concurrent import tqdm_joblib   # 让 joblib 也带总进度条


# === 1. 引入你现有的模型 / 数据 / 常量 ===========================
from init_param import (QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
                        init_pars)
from init_data_point4 import (time_points_train, concentration_data_train,
                              input_dose_train, inject_timelen_train)

# ---------- 微分方程 ----------
@njit(fastmath=True)
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    input_rate = R if t <= T_total else 0
    ydot = np.empty(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) \
            + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) \
            + Kreab * y[4] + input_rate / VPlas
    ydot[1] = QL * (y[0]/VPlas - y[1]/VL/PL) - Kbile * y[1]
    ydot[2] = QK * (y[0]/VPlas - y[2]/VK/PK) - y[0]/VPlas*GFR*Free \
            - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2]/VK/PK)
    ydot[3] = QRest * (y[0]/VPlas - y[3]/VRest/PRest)
    ydot[4] = y[0]/VPlas*GFR*Free + (Vmax_baso*y[2]/VK/PK)/(Km_baso+y[2]/VK/PK) \
            - y[4]*Kurine - Kreab*y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]
    return ydot
def FIT_model(t, D_total, T_total, *params):
    R = D_total / T_total
    y0 = np.zeros(7)
    
    def rhs(ti, yi):
        return derivshiv(yi, ti, params, R, T_total)
    try:
        sol = solve_ivp(rhs, (t[0], t[-1]), y0,
                        t_eval=t, method='LSODA',
                        rtol=1e-6, atol=1e-9,
                        max_step=0.2)
    except Exception:
        return np.full_like(t, np.nan)  # 显式失败值
    if not sol.success or np.any(np.isnan(sol.y[0])) or np.any(np.isinf(sol.y[0])):
        return np.full_like(t, np.nan)  # 明确失败处理

    return sol.y[0] / VPlas

# ---------------------------------------------------------------

# === 2. 读入 0427_Powell 先验中心 ===============================
with open('saved_result/optimized_params0427_Powell.pkl', 'rb') as f:
    theta_start = pickle.load(f)          # 10 维 ndarray

param_names = ["PRest","PK","PL","Kbile","GFR","Free",
               "Vmax_baso","Km_baso","Kurine","Kreab"]
sigma_start   = 0.6   # 给个经验值：残差(log)的SD≈0.5–1.0
theta_start   = np.append(theta_start, sigma_start)   # 变成 11 维
param_names  += ['sigma']  

# === 3. Likelihood（假设残差 ~ N(0, σ²)，σ² 取 1） = -0.5 * RSS ===
def log_likelihood(theta):
    sigma = theta[-1]
    if sigma <= 0:                # 拒绝负 σ
        return -np.inf
    rss = 0.0
    n_tot = 0
    for tp, conc, dose, tinf in zip(time_points_train,
                                    concentration_data_train,
                                    input_dose_train,
                                    inject_timelen_train):
        pred = FIT_model(tp, dose, tinf, *theta[:-1])
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return -np.inf

        # —— 防止 log(0)
        EPS  = 1e-6
        pred = np.clip(pred, EPS, None)
        conc = np.clip(conc, EPS, None)

        diff = np.log(pred) - np.log(conc)
        rss += np.sum(diff**2)
        n_tot += diff.size

    # 同步更新 σ 的先验（半 Cauchy 或 Inv-Gamma）
    return -0.5*rss/sigma**2 - n_tot*np.log(sigma)

# === 4. Proposal：对 log(θ) 做随机游走，高维共线更稳 ==========
theta_log_start = np.log(theta_start)
#step_sizes = 0.005 * np.ones_like(theta_log_start)   # 5% 抖动；可微调
cov_prop = (0.05 ** 2) * np.eye(len(theta_log_start))  
def propose(current_log, rng):
    """多维正态提案。rng 是 numpy.random.Generator。"""
    return current_log + rng.multivariate_normal(
        mean=np.zeros_like(current_log),
        cov=cov_prop
    )

#def propose(current_log):
#    return current_log + np.random.normal(scale=step_sizes)

# === 5. 采样参数 ===============================================
n_iter   = 50000       # 总迭代
burn_in  = 5000       # 丢弃前 burn_in
thin     = 20          # 每 thin 取一次，减少自相关
rng = np.random.default_rng(seed=20240610)
n_chain = 4

# === 6. 单链采样函数 ==============================================
def run_chain(seed,chain_id=None, progress_bar=True):
    rng = np.random.default_rng(seed)
    chain   = np.empty((n_iter, len(theta_start)))
    loglike = np.empty(n_iter)

    curr_log   = theta_log_start.copy()
    curr_theta = theta_start.copy()
    curr_ll    = log_likelihood(curr_theta)
    accept_cnt = 0

    iter_range = range(n_iter)
    if progress_bar:
        desc_txt = f"Chain {chain_id}" if chain_id is not None else "Sampling"
        iter_range = tqdm(iter_range, desc=desc_txt, leave=False)

    for i in iter_range:
        prop_log   = propose(curr_log, rng)
        prop_theta = np.exp(prop_log)
        prop_ll    = log_likelihood(prop_theta)

        if np.log(rng.uniform()) < (prop_ll - curr_ll):
            curr_log, curr_theta, curr_ll = prop_log, prop_theta, prop_ll
            accept_cnt += 1

        chain[i]   = curr_theta
        loglike[i] = curr_ll
        # === 每 1000 步自适应更新协方差 (Haario-2001) ===
        if (i + 1) % 1000 == 0 and i > 0:
            recent = np.log(chain[max(0, i - 999): i + 1])
            emp_cov = np.cov(recent.T)            # 计算最近 1000 样本的协方差
            d = emp_cov.shape[0]
            cov_prop[:] = (2.38 ** 2 / d) * emp_cov + 1e-9 * np.eye(d)

    acc_rate = accept_cnt / n_iter
    return chain, loglike, acc_rate
# === 6b. 多链运行 ================================================
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == "__main__":

    # ========== 并行 + 总体进度条 ==========
    print(f"⏳ 共 {n_chain} 条链，每条 {n_iter} 次迭代，开始并行采样……")

    def _one_chain(cid):
        """子进程跑一条链；关闭内部 tqdm，避免刷屏"""
        return run_chain(seed=20240613 + cid,
                        chain_id=cid + 1,
                        progress_bar=False)

    t0 = time.time()  # 计时开始
    pbar = tqdm(total=n_chain, desc="Chains done", position=0)
    # ---- 并行执行，每完成一个任务就手动 pbar.update() ----
    results = []
    for res in Parallel(n_jobs=n_chain)(
            delayed(_one_chain)(cid) for cid in range(n_chain)):
        results.append(res)
        pbar.update()          # 主进程收到结果后 +1

    pbar.close()
    chain_list, loglike_list, acc_rates = zip(*results)
    print(f"⏱️  全部链完成，用时 {time.time() - t0:,.1f} 秒")
    # ======================================


    # === 7. 后验合并 ===============================================
    post_list   = []
    for c in chain_list:
        post_c = c[burn_in::thin]          # shape = (draws, n_param)
        post_list.append(post_c)

    post_all   = np.concatenate(post_list, axis=0)        # (n_chain*draws, n_param)
    theta_post_mean = post_all.mean(axis=0)

    print("\n后验均值参数：")
    for name, val in zip(param_names, theta_post_mean):
        print(f"{name:<10} {val:>10.4g}")
    # === 7b. 收敛诊断：多链 R-hat ==================================
    import arviz as az

    # 把每条链的 burn / thin 后数组堆成 (chains, draws, n_param)
    draws_per_chain = post_list[0].shape[0]
    posterior_dict = {
        name: np.stack([pc[:, idx] for pc in post_list])   # shape (n_chain, draws)
        for idx, name in enumerate(param_names)
    }

    idata = az.from_dict(posterior=posterior_dict)

    summary = az.summary(idata, var_names=param_names,
                        round_to=4, filter_vars="like")
    az.plot_trace(idata)
    plt.tight_layout()
    plt.savefig("saved_result/mcmc_traceplot0613.png", dpi=300)
    plt.close()
    print("\n=== ArviZ 收敛诊断 ===")
    print(summary[['mean','r_hat','ess_bulk','ess_tail']])

    if (summary['r_hat'] > 1.01).any():
        print("⚠️  存在 r_hat > 1.01，建议延长采样或调步长。")
    else:
        print("✅  r_hat 全部 ≤ 1.01，收敛良好。")


    # === 8. 保存后验均值 ============
    os.makedirs('saved_result', exist_ok=True)
    out_path = 'saved_result/mcmc_params0613.pkl'
    with open(out_path,'wb') as f:
        pickle.dump(theta_post_mean, f)
    print(f"\n🌟 已保存到 {out_path}")

    # ------------------------------------------------------------------
    # === 9. 打印“最终优化参数对比”表 ==================================
    import pandas as pd
    from init_param import init_pars            # ← 这是你脚本里原始基线向量

    df_param = pd.DataFrame({
        '参数': pd.Series(param_names),
        '初始参数值': pd.Series(init_pars),               # 注：若想比 Powell 先验就换成 theta_start
        'MCMC均值':  pd.Series(theta_post_mean)
    })

    print("\n=== 🏆 最终优化参数对比（MCMC）🏆 ===")
    print(df_param.to_string(index=False))
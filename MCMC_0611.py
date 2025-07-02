"""
自写 Metropolis-Hastings 采样器，用 0427_Powell 初步拟合结果做先验中心。
采完链后把后验均值保存为 saved_result/mcmc_params0427.pkl，
可直接被 Simu.py / simu_plot.py 调用。
运行方式:
    python mcmc_metropolis0427.py
"""

import numpy as np
from tqdm import tqdm
import pickle, time, os, datetime,pandas as pd
from scipy.integrate import odeint
from joblib import Parallel, delayed
#from tqdm.contrib.concurrent import tqdm_joblib   # 让 joblib 也带总进度条


# === 1. 引入你现有的模型 / 数据 / 常量 ===========================
from init_param import (QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
                        init_pars)
from init_data_point4 import (time_points_train, concentration_data_train,
                              input_dose_train, inject_timelen_train)

# ---------- 与 modfit0610.py 中一致 ----------
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    input_rate = R if t <= T_total else 0
    ydot = np.zeros(7)
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
    y = odeint(derivshiv, y0, t, args=(params, R, T_total),
               rtol=1e-6, atol=1e-9, h0=0.1)
    return y[:, 0] / VPlas
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
step_sizes = 0.05 * np.ones_like(theta_log_start)   # 5% 抖动；可微调

def propose(current_log):
    return current_log + np.random.normal(scale=step_sizes)

# === 5. 采样参数 ===============================================
n_iter   = 5000       # 总迭代
burn_in  = 1000       # 丢弃前 burn_in
thin     = 5          # 每 thin 取一次，减少自相关
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
        prop_log   = propose(curr_log)
        prop_theta = np.exp(prop_log)
        prop_ll    = log_likelihood(prop_theta)

        if np.log(rng.uniform()) < (prop_ll - curr_ll):
            curr_log, curr_theta, curr_ll = prop_log, prop_theta, prop_ll
            accept_cnt += 1

        chain[i]   = curr_theta
        loglike[i] = curr_ll

    acc_rate = accept_cnt / n_iter
    return chain, loglike, acc_rate
# === 6b. 多链运行 ================================================
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == "__main__":
    chain_list, loglike_list, acc_rates = [], [], []

    # —— 显示一个总进度条
    with tqdm(total=n_chain, desc="Total sampling") as prog:

        def _run_one_chain(cid):
            chain_c, ll_c, acc_c = run_chain(seed=20240611+cid,
                                             chain_id=cid+1,
                                             progress_bar=False)  # 子进程不带 tqdm
            return chain_c, ll_c, acc_c

        results = Parallel(n_jobs=n_chain)(
            delayed(_run_one_chain)(cid) for cid in range(n_chain)
        )

        for cid, (chain_c, ll_c, acc_c) in enumerate(results, 1):
            chain_list.append(chain_c)
            loglike_list.append(ll_c)
            acc_rates.append(acc_c)
            print(f"链 {cid} 完成，接受率 {acc_c:.2f}")
            prog.update(1)

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
    print("\n=== ArviZ 收敛诊断 ===")
    print(summary[['mean','r_hat','ess_bulk','ess_tail']])

    if (summary['r_hat'] > 1.01).any():
        print("⚠️  存在 r_hat > 1.01，建议延长采样或调步长。")
    else:
        print("✅  r_hat 全部 ≤ 1.01，收敛良好。")


    # === 8. 保存后验均值 ============
    os.makedirs('saved_result', exist_ok=True)
    out_path = 'saved_result/mcmc_params0611.pkl'
    with open(out_path,'wb') as f:
        pickle.dump(theta_post_mean, f)
    print(f"\n🌟 已保存到 {out_path}")

    # ------------------------------------------------------------------
    # === 9. 打印“最终优化参数对比”表 ==================================
    import pandas as pd
    from init_param import init_pars            # ← 这是你脚本里原始基线向量

    df_param = pd.DataFrame({
        '参数': pd.Series(param_names),
        '初始参数值': pd.Series(init_pars,               # 注：若想比 Powell 先验就换成 theta_start
        'MCMC均值':  theta_post_mean
    })

    print("\n=== 🏆 最终优化参数对比（MCMC）🏆 ===")
    print(df_param.to_string(index=False))
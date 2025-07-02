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
assert len(theta_start)==10

# === 3. Likelihood（假设残差 ~ N(0, σ²)，σ² 取 1） = -0.5 * RSS ===
def log_likelihood(theta):
    rss = 0.0
    for tp, conc, dose, tinf in zip(time_points_train,
                                    concentration_data_train,
                                    input_dose_train,
                                    inject_timelen_train):
        pred = FIT_model(tp, dose, tinf, *theta)
        rss += np.sum((pred - conc)**2)
    return -0.5 * rss

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

# === 6. MCMC 主循环 ============================================
chain   = np.empty((n_iter, len(theta_start)))
loglike = np.empty(n_iter)

curr_log   = theta_log_start.copy()
curr_theta = theta_start.copy()
curr_ll    = log_likelihood(curr_theta)

accept_cnt = 0
start_time = time.time()
with tqdm(range(n_iter), desc="Sampling") as pbar:
    for i in pbar:
        prop_log   = propose(curr_log)
        prop_theta = np.exp(prop_log)                # 保证正值
        prop_ll    = log_likelihood(prop_theta)

        if np.log(rng.uniform()) < (prop_ll - curr_ll):
            curr_log, curr_theta, curr_ll = prop_log, prop_theta, prop_ll
            accept_cnt += 1

        chain[i]   = curr_theta
        loglike[i] = curr_ll
        if (i+1)%500==0:
            pbar.set_postfix(LL=f"{curr_ll:.1f}", acc=f"{accept_cnt/(i+1):.2f}")

runtime = time.time()-start_time
print(f"\n采样完成，总耗时 {runtime/60:.2f} min，接受率 {accept_cnt/n_iter:.2f}")

# === 7. 后验处理 ===============================================
post_chain = chain[burn_in::thin]
theta_post_mean = post_chain.mean(axis=0)
print("\n后验均值参数：")
for name,val in zip(param_names, theta_post_mean):
    print(f"{name:<10} {val:>10.4g}")

# === 8. 保存后验均值 ============
os.makedirs('saved_result', exist_ok=True)
out_path = 'saved_result/mcmc_params0610.pkl'
with open(out_path,'wb') as f:
    pickle.dump(theta_post_mean, f)
print(f"\n🌟 已保存到 {out_path}")

# ------------------------------------------------------------------
# === 9. 打印“最终优化参数对比”表 ==================================
import pandas as pd
from init_param import init_pars            # ← 这是你脚本里原始基线向量

df_param = pd.DataFrame({
    '参数': param_names,
    '初始参数值': init_pars,               # 注：若想比 Powell 先验就换成 theta_start
    'MCMC均值':  theta_post_mean
})

print("\n=== 🏆 最终优化参数对比（MCMC）🏆 ===")
print(df_param.to_string(index=False))
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
import arviz as az
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
with open('/nfs/home/y18300744/MTXmodel/saved_result/optimized_params0427_Powell.pkl', 'rb') as f:
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
step_sizes = 0.05 * np.ones_like(theta_log_start)   # 5% 抖动；可微调

def propose(current_log, rng):
    return current_log + rng.normal(scale=step_sizes)

# === 5. 采样参数
#  ===============================================
checkpoint_int = 50000          # ★★ 新：每 1万 步存一次 ckpt
resume_path    = ""              # ★★ 新：空字串=不续跑；填路径即续
jit_scale = 0.05        # <-- 初始抖动幅度 (log 空间±5 %)
step_report = 20000
n_iter   = 250000      # 总迭代
burn_in  = 25000       # 丢弃前 burn_in
thin     = 50          # 每 thin 取一次，减少自相关
rng = np.random.default_rng(seed=20240610)
n_chain = 4

# === 6. 单链采样函数 ==============================================
def run_chain(seed,chain_id=None, progress_bar=True):
    rng = np.random.default_rng(seed)
    chain   = np.empty((n_iter, len(theta_start)))
    loglike = np.empty(n_iter)

     # 若填写了 resume_path，则覆盖初始点 & RNG 状态 ------------------
    if chain_id == 1 and resume_path and os.path.exists(resume_path):  # ★★
        ckpt = pickle.load(open(resume_path, "rb"))
        curr_log = ckpt["phi"]
        rng.bit_generator.state = ckpt["rng"]
        start_iter = ckpt["iter"]
        print(f"▶️  从步 {start_iter} 续跑 (Chain 1)")
    else:
        curr_log = theta_log_start + rng.normal(scale=jit_scale)
        start_iter = 0
 
    #curr_log   = theta_log_start + rng.normal(scale=jit_scale)#★★ 修改 
    curr_theta = np.exp(curr_log)
    curr_ll    = log_likelihood(curr_theta)
    accept_cnt = 0

    iters = tqdm(range(start_iter, start_iter + n_iter),   # ★★
                  desc=f"Chain {chain_id}", leave=False)
    for i in iters:
        prop_log   = propose(curr_log, rng)#★★ 修改 
        prop_theta = np.exp(prop_log)
        prop_ll    = log_likelihood(prop_theta)

        if np.log(rng.random()) < (prop_ll - curr_ll):#★★ 修改 
            curr_log, curr_theta, curr_ll = prop_log, prop_theta, prop_ll
            accept_cnt += 1

        idx = i - start_iter
        chain[idx]   = curr_theta
        loglike[idx] = curr_ll

        # === ★★ checkpoint：每 10 k 步 dump =====================
        if (i + 1) % checkpoint_int == 0:
            ck_path = f"ckpt_chain{chain_id}_{i+1}.pkl"
            pickle.dump({
                "iter":   i + 1,
                "phi":    curr_log,
                "theta":  curr_theta,
                "ll":     curr_ll,
                "rng":    rng.bit_generator.state,
            }, open(ck_path, "wb"))
            print(f"💾 已保存 {ck_path}")

    acc_rate = accept_cnt / n_iter
    ll_mean  = loglike[burn_in:].mean()        # <<< PATCH ③
    return chain, loglike, acc_rate,ll_mean
# === 6b. 多链运行 ================================================
from tqdm import tqdm
# ==== ★ 新增：并行采样所需 ====
from joblib import Parallel, delayed          # ★

if __name__ == "__main__":
    results = Parallel(n_jobs=n_chain, prefer="processes")(    # <<< PATCH ④
        delayed(run_chain)(seed=20240611 + cid, chain_id=cid+1)
        for cid in range(n_chain)
    )
    chain_list, loglike_list, acc_rates = [], [], []
    for cid, (c, ll, acc, llm) in enumerate(results, 1):       # <<< PATCH ⑤
        chain_list.append(c); loglike_list.append(ll); acc_rates.append(acc)
        print(f"链 {cid}: 接受率={acc:.2f}, LL均值={llm:.1f}")
    # === ★ 结果解包 =================================================
   
    # === ★ 打印每条链指标并一次性解包 ===============================
    for cid, (c, ll, acc, llm) in enumerate(results, 1):
        print(f"链 {cid}: 接受率={acc:.2f}, LL均值={llm:.1f}")

    chain_list, loglike_list, acc_rates, ll_means = map(list, zip(*results))
    print("\n采样完成，各链接受率:", acc_rates)

# # === 8. 分段诊断表  (行号≈260) ===================================
# report_rows = []
# for draws in range(checkpoint_int, n_iter+1, checkpoint_int):
#     posterior = {
#         name: np.stack([c[:draws, idx] for c in chain_list])
#         for idx, name in enumerate(param_names)
#     }
#     idata = az.from_dict(posterior=posterior)
#     summ  = az.summary(idata, var_names=param_names, filter_vars="like")
#     max_rhat = float(summ['r_hat'].max())
#     min_ess  = int(summ['ess_bulk'].min())

   
#     report_rows.append(dict(Draws=draws, acc=np.mean(acc_rates),
#                             max_rhat=max_rhat, min_ess=min_ess
#                             ))
#     print(f"📊 {draws:,} draws | r̂={max_rhat:.3f} | ESS={min_ess} ")
#     # === 7. 后验合并 ===============================================
    # === 7. 后验合并 & R-hat 初判 ===================================
    post_list = [c[burn_in::thin] for c in chain_list]

    import pickle, os
    os.makedirs('/nfs/home/y18300744/MTXmodel/saved_result', exist_ok=True)

    for k, pc in enumerate(post_list, 1):          # post_list 里是 burn-in 后样本
        theta_k = pc.mean(axis=0)                  # 单链后验均值
            # ✅ 保存该链 burn-in 后的完整后验样本
        draw_path = f"/nfs/home/y18300744/MTXmodel/saved_result/chain{k}_draws0628.pkl"
        with open(draw_path, "wb") as f:
            pickle.dump(pc, f)
        print(f"链 {k} 后验样本已保存 ➜ {draw_path}")
        
        # ★ 新增：格式化打印到终端 ★
        print(f"\n后验均值参数 — 链 {k}")
        for name, val in zip(param_names, theta_k):
            print(f"{name:<12s}{val:>12.6g}")
        path_k  = f"/nfs/home/y18300744/MTXmodel/saved_result/chain{k}_params0628.pkl"
        pickle.dump(theta_k, open(path_k, "wb"))   # 保存
        print(f"链 {k} 后验均值已保存 ➜ {path_k}")
    # =========================================================

    post_all  = np.concatenate(post_list, axis=0)               # ★ 新增
    theta_post_mean = post_all.mean(axis=0)                     # ★ 新增
    draws_per_chain = post_list[0].shape[0]
    posterior_dict = {
        name: np.stack([pc[:, idx] for pc in post_list])
        for idx, name in enumerate(param_names)
    }
    import arviz as az

# === PATCH-1 : 并行采样不变，得到 results 后继续 ===
# ↓↓↓ 这段在原脚本 print 汇总之后插入 ↓↓↓
# ------------------------------------------------------------
import arviz as az, pandas as pd, os, datetime
report_rows = []                              # 存迭代-诊断汇总

# —— 把四条链拼成 (chain, draw, dim) 方便分段分析 ——
chains_np = np.stack(chain_list)              # shape = (4 , n_iter , 11)

for k in range(step_report, n_iter + 1, step_report):
    # === PATCH-2 : 构造 ArviZ InferenceData =========
    post_slice = [c[:k] for c in chain_list]          # 每链前 k draws
    posterior   = {
        name: np.stack([pc[:, idx] for pc in post_slice])
        for idx, name in enumerate(param_names)
    }
    idata_k = az.from_dict(posterior=posterior)

    summ_k   = az.summary(idata_k, var_names=param_names,
                          round_to=4, filter_vars="like")
    max_rhat = summ_k['r_hat'].max()
    min_ess  = summ_k['ess_bulk'].min()

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    report_rows.append(dict(
        时刻=ts, Draws=k, max_rhat=float(max_rhat), min_ess=int(min_ess)))

    print(f"\n📊 迭代 {k:,}:  max r̂={max_rhat:.3f},  min ESS={min_ess:.0f}")



    # idata   = az.from_dict(posterior=posterior_dict)
    # summary = az.summary(idata, var_names=param_names,
    #                      round_to=4, filter_vars="like")       # <<< PATCH ⑥

    # print("\n=== ArviZ 收敛诊断 (R-hat 初判) ===")
    # print(summary[['mean','r_hat','ess_bulk','ess_tail']])
    # # ----------------------------------------------------------------

    # print("\n=== ArviZ 收敛诊断 ===")
    # print(summary[['mean','r_hat','ess_bulk','ess_tail']])

    # if (summary['r_hat'] > 1.01).any():
    #     print("⚠️  存在 r_hat > 1.01，建议延长采样或调步长。")
    # else:
    #     print("✅  r_hat 全部 ≤ 1.01，收敛良好。")
    # az.plot_trace(idata)

    # === PATCH-3 : 写 Excel ===============================
    rep_df = pd.DataFrame(report_rows)
    out_xlsx = "/nfs/home/y18300744/MTXmodel/saved_result/mcmc_diag_0628.xlsx"
    rep_df.to_excel(out_xlsx, index=False)
    print(f"\n✅ 诊断表已保存 ➜ {out_xlsx}")
    # ------------------------------------------------------------


    plt.tight_layout()
    plt.savefig("/nfs/home/y18300744/MTXmodel/saved_result/mcmc_traceplot0628.png", dpi=300)
    plt.close()

    # === 8. 保存后验均值 ============
    #os.makedirs('saved_result', exist_ok=True)
    out_path = '/nfs/home/y18300744/MTXmodel/saved_result/mcmc_params0628.pkl'
    with open(out_path,'wb') as f:
        pickle.dump(theta_post_mean, f)
    print(f"\n🌟 已保存到 {out_path}")

    # ------------------------------------------------------------------
    # === 9. 打印“最终优化参数对比”表 ==================================
    import pandas as pd
    from init_param import init_pars            # ← 这是你脚本里原始基线向量

    df_param = pd.DataFrame({
        '参数': pd.Series(param_names),
        '初始参数值': pd.Series(theta_start),               # 注：若想比 Powell 先验就换成 theta_start
        'MCMC均值':  pd.Series(theta_post_mean)
    })

    print("\n=== 🏆 最终优化参数对比（MCMC）🏆 ===")
    print(df_param.to_string(index=False))
# ========= 文件名: modmcmc0509_ident.py =========
import pymc3 as pm, theano.tensor as tt, numpy as np, pickle, os
from tqdm import tqdm
from init_param import init_pars, QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
from init_data_point4 import (time_points_train, concentration_data_train,
                              input_dose_train, inject_timelen_train)
#from modmcmc0509 import derivshiv                       # 复用 ODE

# --- 固定量 ----
PRest_val   = init_pars['PRest']
Free_val    = 0.5                                       # 固定
Km_baso_val = init_pars['Km_baso']                      # 参与 CL_baso

# ---------- modmcmc0509.py ----------
import theano.tensor as tt
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas

def derivshiv(y, t, theta, R, T_total):
    """
    Pure-symbolic RHS for DifferentialEquation.
    theta: length-10 tensor (PRest, PK, …, Kreab)
    """
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = [
        theta[i] for i in range(10)
    ]

    input_rate = tt.switch(tt.le(t, T_total), R, 0.0)

    dot0 = (QRest*y[3]/VRest/PRest) + (QK*y[2]/VK/PK) + (QL*y[1]/VL/PL) \
         - (QPlas*y[0]/VPlas) + Kreab*y[4] + input_rate/VPlas
    dot1 = QL*(y[0]/VPlas - y[1]/VL/PL) - Kbile*y[1]
    dot2 = QK*(y[0]/VPlas - y[2]/VK/PK) \
         - y[0]/VPlas*GFR*Free \
         - (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK)
    dot3 = QRest*(y[0]/VPlas - y[3]/VRest/PRest)
    dot4 = y[0]/VPlas*GFR*Free \
         + (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK) \
         - y[4]*Kurine - Kreab*y[4]
    dot5 = Kurine*y[4]
    dot6 = Kbile*y[1]

    # 保证返回 rank-1 Tensor
    vec = tt.stack([dot0, dot1, dot2, dot3, dot4, dot5, dot6], axis=0)
    return tt.reshape(vec, (7,))


# --- 开始建模 ---
with pm.Model() as model:
    # 1) 可识别采样参数 (6)
    baseline_clnet = init_pars['Kurine'] - init_pars['Kreab']
    if baseline_clnet <= 0:
        baseline_clnet = 1e-6   # 兜底正数

    CL_net = pm.Lognormal(
        'CL_net',
        mu=np.log(baseline_clnet),
        sigma=0.5,              # 给宽一点
        testval=baseline_clnet
    )
        #CL_net      = pm.Lognormal('CL_net',  mu=np.log(init_pars['Kurine']-init_pars['Kreab']), sigma=0.3)
    Frac_reabs  = pm.Beta('Frac_reabs', alpha=2, beta=2)               # 0–1
    Kbile       = pm.Lognormal('Kbile',  mu=np.log(init_pars['Kbile']), sigma=0.3)
    PK          = pm.Lognormal('PK',     mu=np.log(init_pars['PK']),    sigma=0.25)
    PL          = pm.Lognormal('PL',     mu=np.log(init_pars['PL']),    sigma=0.25)
    #  GFR 可选：采样 or 固定
    GFR         = pm.Lognormal('GFR',    mu=np.log(init_pars['GFR']),   sigma=0.5)

    # 2) 由派生量恢复原参数
    Kurine = pm.Deterministic('Kurine', CL_net / (1 - Frac_reabs))
    Kreab  = pm.Deterministic('Kreab',  Kurine * Frac_reabs)
    CL_baso = pm.Lognormal('CL_baso', mu=np.log(init_pars['Vmax_baso']/Km_baso_val), sigma=0.4)
    Vmax_baso = pm.Deterministic('Vmax_baso', CL_baso * Km_baso_val)

    # 3) 常量 → theano tensor
    PRest = tt.as_tensor_variable(PRest_val)
    Free  = tt.as_tensor_variable(Free_val)
    Km_baso = tt.as_tensor_variable(Km_baso_val)

    # 4) 打包成 10 维参数向量（顺序别改）
    theta = tt.stack([
        PRest, PK, PL, Kbile, GFR,
        Free,  Vmax_baso, Km_baso,
        Kurine, Kreab
    ])

    sigma = pm.HalfNormal('sigma', 1.)
    def ode_fun(y, t_, theta_):
        return derivshiv(y, t_, theta_, R, tlen)
    # ---- 观测循环 ----
    for i in tqdm(range(len(time_points_train)), desc="likelihood"):
        t = time_points_train[i].astype('float64')
        dose = input_dose_train[i]
        tlen = inject_timelen_train[i]
        R = dose / tlen

        y_hat = pm.ode.DifferentialEquation(
            func=ode_fun,
            times=t,
            n_states=7,
            n_theta=10,
            t0=0,
            jacobian=False 
            #args=(R, tlen)
        )
        y_hat = y_hat (y0=np.zeros(7), theta=theta)[:, 0] / VPlas
        pm.Normal(f'obs_{i}', mu=y_hat, sigma=sigma,
                  observed=concentration_data_train[i])

    # ---- 采样器：NUTS（grad free via Op; fallback DEMetropolisZ） ----
    step = pm.DEMetropolisZ()
    trace = pm.sample(2500, tune=1500, step=step,
                      chains=4, cores=4, random_seed=1,
                      return_inferencedata=False, progressbar=True)

    print(pm.summary(trace))
    pm.save_trace(trace, directory='saved_result/trace_ident', overwrite=True)
# ===============================================
# ------- 5. 打印 & 保存优化参数 ---------------------------------
import pandas as pd, pickle, os, numpy as np

param_means = {
    'CL_net'     : trace['CL_net'    ].mean(),
    'Frac_reabs' : trace['Frac_reabs'].mean(),
    'Kbile'      : trace['Kbile'     ].mean(),
    'PK'         : trace['PK'        ].mean(),
    'PL'         : trace['PL'        ].mean(),
    'GFR'        : trace['GFR'       ].mean(),          # 若固定则值恒定
    'sigma'      : trace['sigma'     ].mean()
}
print("\n★ Posterior means")
for k,v in param_means.items():
    print(f"{k:12s}: {v:>10.4g}")

# 保存
os.makedirs('saved_result', exist_ok=True)
pd.Series(param_means).to_csv('saved_result/best_params_ident.csv')
with open('saved_result/best_params_ident.pkl', 'wb') as f:
    pickle.dump(param_means, f)
print("\n✓ 参数已写入 saved_result/best_params_ident.[csv|pkl]")
# modmcmc0509_sens.py  ——  基于敏感性分析的精简版 MCMC
import matplotlib.pyplot as plt
import datetime, os, pickle
from functools import partial

import theano
import theano.tensor as tt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tqdm import tqdm

import pymc3 as pm
import arviz as az

# ------- 本地模块 -------
from init_param import (QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
                        init_pars)
from init_data_point4 import (df, time_points_train, concentration_data_train,
                              input_dose_train, inject_timelen_train)

os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'

# -------------------------------------------------------------------
# 1. ODE 系统
# -------------------------------------------------------------------
def derivshiv(y, t, parms, R, T_total):
    """
    7-室 PBPK 方程（保持原来顺序不变）
    parms 顺序:
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab
    """
    (PRest, PK, PL, Kbile, GFR,
     Free, Vmax_baso, Km_baso, Kurine, Kreab) = parms

    input_rate = R if t <= T_total else 0
    ydot = np.zeros(7)

    ydot[0] = (QRest * y[3] / VRest / PRest) \
              + (QK * y[2] / VK / PK) \
              + (QL * y[1] / VL / PL) \
              - (QPlas * y[0] / VPlas) \
              + Kreab * y[4] \
              + input_rate / VPlas

    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]

    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) \
              - y[0] / VPlas * GFR * Free \
              - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)

    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)

    ydot[4] = y[0] / VPlas * GFR * Free \
              + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) \
              - y[4] * Kurine - Kreab * y[4]

    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]
    return ydot


# -------------------------------------------------------------------
# 2. theano 包装 (接口保持 10 参不变)
# -------------------------------------------------------------------
'''
@as_op 装饰器：告诉 Theano “这是一个不可求梯度的黑盒函数”。

itypes：声明输入类型列表。前 3 个是时间向量、总剂量、给药时长；后面是 10 个标量参数。

otypes：输出是 1-D 向量（血浆浓度随时间）。

为什么要包起来？ PyMC3 的计算图必须由 Theano 张量组成。把 odeint 包进 Op 后，就能在 PyMC3 模型里当作确定性节点，虽不能自动求导，但与随机参数共同参与 Monte-Carlo 采样。

缺点：因为无梯度，NUTS/HMC 之类基于梯度的方法不可用，故脚本后面显式指定随机步进器 DEMetropolisZ。
'''
@theano.compile.ops.as_op(
    itypes=[tt.dvector, tt.dscalar, tt.dscalar] + [tt.dscalar]*10,
    otypes=[tt.dvector]
)
def theano_FIT_model(t, D_total, T_total,
                     PRest, PK, PL, Kbile, GFR, Free,
                     Vmax_baso, Km_baso, Kurine, Kreab):

    R = D_total / T_total
    y0 = np.zeros(7)
    params = [PRest, PK, PL, Kbile, GFR,
              Free, Vmax_baso, Km_baso, Kurine, Kreab]

    y = odeint(derivshiv, y0, t,
               args=(params, R, T_total),
               rtol=1e-4, atol=1e-6, h0=0.1)
    return y[:, 0] / VPlas


# -------------------------------------------------------------------
# 3. 读取 Powell 优化结果（作为基线）
# -------------------------------------------------------------------
with open('saved_result/optimized_params0427_Powell.pkl', 'rb') as f:
    fit_params = pickle.load(f)

# 将低敏感度参数固定为 Powell 值
PRest_c   = tt.as_tensor_variable(fit_params[0])# ← as_tensor_variable 把常数塞进 Theano 图
Free_c    = tt.as_tensor_variable(fit_params[5])
Km_baso_c = tt.as_tensor_variable(fit_params[7])
Kurine_c  = tt.as_tensor_variable(fit_params[8])
Kreab_c   = tt.as_tensor_variable(fit_params[9])

# -------------------------------------------------------------------
# 4. MCMC
# -------------------------------------------------------------------
if __name__ == '__main__':
    with pm.Model() as model:

        # ===== 仅为高敏感参数设定先验 =====
        PK      = pm.Lognormal('PK',     mu=np.log(fit_params[1]), sigma=0.3)
        PL      = pm.Lognormal('PL',     mu=np.log(fit_params[2]), sigma=0.3)
        Kbile   = pm.Lognormal('Kbile',  mu=np.log(fit_params[3]), sigma=0.3)
        GFR     = pm.Lognormal('GFR',    mu=np.log(fit_params[4]), sigma=0.6)
        Vmax_baso = pm.Lognormal('Vmax_baso',
                                 mu=np.log(fit_params[6]), sigma=0.4)

        # 共用观测噪声
        sigma = pm.HalfNormal("sigma", 1)

        # 逐个病人建观测
        for i in tqdm(range(len(time_points_train)), desc='Building likelihood'):
            t_pts   = tt.as_tensor_variable(time_points_train[i].astype(np.float64))
            D_total = tt.as_tensor_variable(input_dose_train[i].astype(np.float64))
            T_total = tt.as_tensor_variable(inject_timelen_train[i].astype(np.float64))
            conc_obs = concentration_data_train[i]

            mu = theano_FIT_model(
                t_pts, D_total, T_total,
                PRest_c, PK, PL, Kbile, GFR, Free_c,
                Vmax_baso, Km_baso_c, Kurine_c, Kreab_c
            )
            pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=conc_obs)

        # --- 采样 ---
        step = pm.DEMetropolisZ()   # ← 明确指定采样器
        trace = pm.sample(
            draws=2000,
            tune=10000,
            step=step,           # NUTS target_accept
            chains=4, cores=4,
            random_seed=1,
            progressbar=True,
            return_inferencedata=False 
        )

        # --- 结果输出 ---
        summ = pm.summary(trace)
        print(summ)

        # 组装完整 10 维参数（5 随机 + 5 固定）
        best_params = [
            float(PRest_c.eval()),                    # PRest 固定
            summ.loc['PK',         'mean'],
            summ.loc['PL',         'mean'],
            summ.loc['Kbile',      'mean'],
            summ.loc['GFR',        'mean'],
            float(Free_c.eval()),                     # Free 固定
            summ.loc['Vmax_baso',  'mean'],
            float(Km_baso_c.eval()),
            float(Kurine_c.eval()),
            float(Kreab_c.eval())
        ]

        os.makedirs('saved_result', exist_ok=True)
        with open('saved_result/best_params0511_sens.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        print("✔ MCMC 优化完成，参数已保存 → saved_result/best_params0511_sens.pkl")

        # 诊断图
        pm.traceplot(trace)
        plt.savefig('trace0511_sens.svg', dpi=300)

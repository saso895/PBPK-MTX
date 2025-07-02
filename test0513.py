# -*- coding: utf-8 -*-
"""
modmcmc0513_log_fix.py  ——  修掉 deriv() 参数不匹配
"""
import numpy as np, pymc3 as pm, theano.tensor as tt, arviz as az
from functools import partial
from init_param import (QRest, QK, QL, QPlas, VRest, VK, VL, VPlas)
from init_data_point4 import (time_points_train, concentration_data_train,
                              input_dose_train, inject_timelen_train)
import pickle, os

# ---------- 1  仍然保持 10 个 theta ----------
def _deriv_core(y, t, theta, R, T_total):
    """真正的 ODE；theta 已经是 len==10 的向量"""
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = theta
    input_rate = tt.switch(tt.le(t, T_total), R, 0.0)

    ydot = tt.zeros((7,))
    ydot = tt.set_subtensor(ydot[0],
        (QRest * y[3] / VRest / PRest)
      + (QK   * y[2] / VK   / PK)
      + (QL   * y[1] / VL   / PL)
      - (QPlas* y[0] / VPlas)
      + Kreab * y[4]
      + input_rate / VPlas)
    ydot = tt.set_subtensor(ydot[1], QL*(y[0]/VPlas - y[1]/VL/PL) - Kbile*y[1])
    ydot = tt.set_subtensor(ydot[2],
        QK*(y[0]/VPlas - y[2]/VK/PK)
      - y[0]/VPlas*GFR*Free
      - (Vmax_baso * y[2]/VK/PK) / (Km_baso + y[2]/VK/PK))
    ydot = tt.set_subtensor(ydot[3], QRest*(y[0]/VPlas - y[3]/VRest/PRest))
    ydot = tt.set_subtensor(ydot[4],
        y[0]/VPlas*GFR*Free
      + (Vmax_baso * y[2]/VK/PK) / (Km_baso + y[2]/VK/PK)
      - y[4]*Kurine - Kreab*y[4])
    ydot = tt.set_subtensor(ydot[5], Kurine*y[4])
    ydot = tt.set_subtensor(ydot[6], Kbile*y[1])
    return ydot

# ---------- 2  读 Powell 初值 ----------
fit_params = pickle.load(open('saved_result/optimized_params0427_Powell.pkl', 'rb'))

with pm.Model() as model:

    # -------- 2.1   先验（同上，略） --------
    PRest, Free, Km_baso, Kurine, Kreab = [tt.as_tensor_variable(v) for v in
        (fit_params[0], fit_params[5], fit_params[7], fit_params[8], fit_params[9])]

    log_PK   = pm.Normal ('log_PK',   mu=np.log(fit_params[1]), sigma=0.3)
    log_PL   = pm.Normal ('log_PL',   mu=np.log(fit_params[2]), sigma=0.3)
    log_Kbile= pm.Normal ('log_Kbile',mu=np.log(fit_params[3]), sigma=0.3)
    GFR      = pm.TruncatedNormal('GFR', mu=fit_params[4], sigma=3, lower=1.0)
    log_Vmax = pm.Normal ('log_Vmax', mu=np.log(fit_params[6]), sigma=0.4)

    PK        = pm.Deterministic('PK', tt.exp(log_PK))
    PL        = pm.Deterministic('PL', tt.exp(log_PL))
    Kbile     = pm.Deterministic('Kbile', tt.exp(log_Kbile))
    Vmax_baso = pm.Deterministic('Vmax_baso', tt.exp(log_Vmax))
    sigma     = pm.HalfNormal('sigma', 1)

    # -------- 2.2   受试者循环 --------
    for i in range(len(time_points_train)):
        t_pts   = time_points_train[i]
        R_i     = input_dose_train[i] / inject_timelen_train[i]
        T_i     = inject_timelen_train[i]
        y0      = np.zeros(7)

        # 让 DifferentialEquation 只看到 3 个参数
        deriv_i = lambda y, t, theta, R=R_i, T_total=T_i: _deriv_core(y, t, theta, R, T_total)

        ode_i = pm.ode.DifferentialEquation(
            func=deriv_i,
            times=t_pts,          # 直接用观测点即可
            n_states=7,
            n_theta=10,
            t0=0.)

        theta_vec = tt.stack([
            PRest, PK, PL, Kbile, GFR, Free,
            Vmax_baso, Km_baso, Kurine, Kreab])

        mu = ode_i(y0=y0, theta=theta_vec)[:, 0] / VPlas
        pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma,
                  observed=concentration_data_train[i])

    # -------- 2.3   NUTS 采样 --------
    trace = pm.sample(draws=3000, tune=2000, target_accept=0.9,
                      init='adapt_diag', chains=4, cores=4, random_seed=2)

    az.to_netcdf(trace, 'trace0513_fix.nc')
    print(az.summary(trace, var_names=['PK','PL','Kbile','GFR','Vmax_baso','sigma']))

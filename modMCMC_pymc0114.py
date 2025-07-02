import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
# from pk_model1209 import pk_model, derivshiv, VPlas
# from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import pickle
import warnings
import theano.tensor as tt
import theano
from functools import partial
from init_param import init_pars
import numpy as np
from scipy.integrate import odeint
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import theano.tensor as tt
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import os
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train

# 获取 CPU 内核数
num_cores = os.cpu_count()
print(f"可用的 CPU 内核数: {num_cores}")

# warnings.filterwarnings('ignore')
theano.config.cxx = 'g++'

# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, log_params):
    '''定义微分方程的函数，包含药物点滴输入'''

    PRest=log_params[0]
    PK=log_params[1]
    PL=log_params[2]
    Kbile=log_params[3]
    GFR=log_params[4]
    Free=log_params[5]
    Vmax_baso=log_params[6]
    Km_baso=log_params[7]
    Kurine = log_params[8]
    Kreab=log_params[9]

    # 确保 input_rate 是标量
    #input_rate = R if t <= T_total else 0
    input_rate = tt.switch(t <= T_total, R, 0)
    ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (
                QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (
                Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[
        4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]

    return ydot

PRest = init_pars["PRest"]
PK = init_pars["PK"]
PL = init_pars["PL"]
Kbile = init_pars["Kbile"]
GFR = init_pars["GFR"]
Free = init_pars["Free"]
Vmax_baso = init_pars["Vmax_baso"]
Km_baso = init_pars["Km_baso"]
Kurine = init_pars["Kurine"]
Kreab = init_pars["Kreab"]
#
T_total = 0
R = 0
# 使用 pymc3 进行 MCMC 采样优化参数
with pm.Model() as model:
    # 设置参数的先验分布
    PRest = pm.Normal('PRest', mu=PRest, sigma=1)
    PK = pm.Normal('PK', mu=PK, sigma=1)
    PL = pm.Normal('PL', mu=PL, sigma=1)
    Kbile = pm.Normal('Kbile', mu=Kbile, sigma=1)
    GFR = pm.Normal('GFR', mu=GFR, sigma=1)
    Free = pm.Normal('Free', mu=Free, sigma=0.1)
    Vmax_baso = pm.Normal('Vmax_baso', mu=Vmax_baso, sigma=10)
    Km_baso = pm.Normal('Km_baso', mu=Km_baso, sigma=1)
    Kurine = pm.Normal('Kurine', mu=Kurine, sigma=0.01)
    Kreab = pm.Normal('Kreab', mu=Kreab, sigma=0.01)

    log_params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]

    # 遍历每组数据，逐组计算拟合值并添加观测数据的似然函数
    for i in range(len(time_points_train)):
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen
        R = D_total / T_total

        # 使用 partial 将 R 和 T_total 作为额外参数传递给 derivshiv
        # ode_func_partial = partial(derivshiv, R=R, T_total=T_total)

        # 定义新的 ODE 求解器
        ode_partial = DifferentialEquation(
            func=derivshiv,
            times=time,
            n_states=7,
            n_theta=10,
            t0=0
        )

        # 求解 ODE
        y = ode_partial(y0=tt.zeros(7), theta=log_params)

        # 计算预测浓度
        VPlas = tt.constant(VPlas)
        mu = y[:, 0] / VPlas
        
        # 使用 pm.Deterministic 来保存每一组预测值
        pm.Deterministic(f'predicted_concentration_{i}', mu)

        y_obs = pm.Normal(f'y_obs_{i}', mu=mu, sigma=0.1, observed=concentration)


    # 使用 NUTS 采样算法进行 MCMC 采样
    #trace = pm.sample(1000, tune=1000, cores=2)
    trace = pm.sample(2000, tune=1000, target_accept=0.9, cores=num_cores, chains=num_cores, step=pm.NUTS(), progressbar=True)
# 将结果转换为 ArviZ 数据结构
    data = az.from_pymc3(trace=trace)

# 检查链的混合情况
az.plot_trace(data)

# 计算总结统计量
summary = az.summary(data)
print(summary)

# 可视化后验分布
az.plot_posterior(data)
# 绘制 MCMC 采样后的结果
pm.traceplot(trace)
plt.show()


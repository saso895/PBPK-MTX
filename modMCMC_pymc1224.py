import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal, Metropolis, sample, find_MAP
from pk_model1209 import pk_model, derivshiv, VPlas
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import pickle
import warnings
import theano.tensor as tt
import theano
from functools import partial
from multiprocessing import Pool, cpu_count
from init_param import init_pars

# warnings.filterwarnings('ignore')
theano.config.cxx = 'g++'

with open('pars/modfit_pars.pkl', 'rb') as f:
    average_params = pickle.load(f)

# 计算优化参数的平均值，作为 MCMC 的初始均值
#average_params = np.mean(optimized_params, axis=0)
#print(f"平均优化参数: {average_params}")
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

# 使用 pymc3 进行 MCMC 采样优化参数
with pm.Model() as model:
    # 设置参数的先验分布
    PRest = pm.Normal('PRest', mu=PRest, sigma=1)
    PK = pm.Normal('PK', mu=PK, sigma=1)
    PL = pm.Normal('PL', mu=PL, sigma=1)
    Kbile = pm.Normal('Kbile', mu=Kbile , sigma=1)
    GFR = pm.Normal('GFR', mu=GFR , sigma=1)
    Free = pm.Normal('Free', mu=Free, sigma=0.1)
    Vmax_baso = pm.Normal('Vmax_baso', mu=Vmax_baso, sigma=10)
    Km_baso = pm.Normal('Km_baso', mu=Km_baso, sigma=1)
    Kurine = pm.Normal('Kurine', mu=Kurine, sigma=0.01)
    Kreab = pm.Normal('Kreab', mu= Kreab, sigma=0.01)

    # 将参数包装成 Theano 的张量
    #log_params = tt.as_tensor_variable([PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab])
    log_params = tt.stack([PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab])
    #params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]


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
        ode_func_partial = partial(derivshiv, R=R, T_total=T_total)

        # 定义 ODE 函数时，接受额外的参数
        def ode_func(y, t, p):
            return derivshiv(y, t, p)


        # 定义新的 ODE 求解器
        ode_partial = pm.ode.DifferentialEquation(
            func=ode_func,
            times=time,
            n_states=7,
            n_theta=10,
            t0=0
        )

        # 求解 ODE
        y = ode_partial(y0=tt.zeros(7), theta=log_params)  # 修改了这行代码
        #y = ode_partial(y0=tt.zeros(7), theta=log_params, extra_args={'R': R, 'T_total': T_total})

        # 计算预测浓度
        VPlas = tt.constant(VPlas)
        mu = y[:, 0] / VPlas
        
        # 使用 pm.Deterministic 来保存每一组预测值
        pm.Deterministic(f'predicted_concentration_{i}', mu)

        y_obs = pm.Normal(f'y_obs_{i}', mu=mu, sigma=0.1, observed=concentration)

        # 初始化 Metropolis-Hastings 采样器
        step = Metropolis(S=cov0)

        # 定义 MCMC 迭代次数
        niter = 10000
        burnin = 3000
        thin = 1

        # 注册并行计算
        num_chains = 4
        cores = cpu_count()

        # 运行 MCMC 采样
        start_time = time.time()
        trace = sample(draws=niter, step=step, init='adapt_diag', cores=cores, chains=num_chains, tune=burnin,
                       discard_tuned_samples=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

# 绘制 MCMC 采样后的结果
pm.traceplot(trace)
plt.show()

# 打印后验分布的摘要
print(pm.summary(trace))

# 生成后验预测值
posterior_predictive = pm.sample_posterior_predictive(trace)

# 绘制观测值和后验预测值
for i in range(len(time_points_train)):
    plt.plot(time_points_train[i], concentration_data_train[i], 'o', label='Observed Data')
    plt.plot(time_points_train[i], posterior_predictive[f'predicted_concentration_{i}'].mean(axis=0), label='Posterior Predictive Mean')
    plt.fill_between(time_points_train[i],
                     posterior_predictive[f'predicted_concentration_{i}'].mean(axis=0) - 2 * posterior_predictive[f'predicted_concentration_{i}'].std(axis=0),
                     posterior_predictive[f'predicted_concentration_{i}'].mean(axis=0) + 2 * posterior_predictive[f'predicted_concentration_{i}'].std(axis=0),
                     alpha=0.3, label='95% HPD Interval')
    plt.legend()
    plt.title(f'Concentration Data and Posterior Predictive for Group {i+1}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pk_model import pk_model
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import pickle
import warnings
import theano.tensor as tt
import theano


# warnings.filterwarnings('ignore')
theano.config.cxx = 'g++'

with open('pars/modfit_pars.pkl', 'rb') as f:
    average_params = pickle.load(f)

# 计算优化参数的平均值，作为 MCMC 的初始均值
#average_params = np.mean(optimized_params, axis=0)
#print(f"平均优化参数: {average_params}")

# 使用 pymc3 进行 MCMC 采样优化参数
with pm.Model() as model:
    # 设置参数的先验分布
    PRest = pm.Normal('PRest', mu=average_params[0], sigma=1)
    PK = pm.Normal('PK', mu=average_params[1], sigma=1)
    PL = pm.Normal('PL', mu=average_params[2], sigma=1)
    Kbile = pm.Normal('Kbile', mu=average_params[3], sigma=1)
    GFR = pm.Normal('GFR', mu=average_params[4], sigma=1)
    Free = pm.Normal('Free', mu=average_params[5], sigma=0.1)
    Vmax_baso = pm.Normal('Vmax_baso', mu=average_params[6], sigma=10)
    Km_baso = pm.Normal('Km_baso', mu=average_params[7], sigma=1)
    Kurine = pm.Normal('Kurine', mu=average_params[8], sigma=0.01)
    Kreab = pm.Normal('Kreab', mu=average_params[9], sigma=0.01)
    
    params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]

    # 使用 Theano 的包装器将解微分方程结果作为观测值
    # observed_conc = pm.Deterministic('observed_conc', pk_model(params))
    predicted_concentrations = []

    # 遍历每组数据，逐组计算拟合值并添加观测数据的似然函数
    for i in range(len(time_points_train)):
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen
        mu = pk_model(time, dose, timelen, *params)
        
        # 使用 pm.Deterministic 来保存每一组预测值
        pm.Deterministic(f'predicted_concentration_{i}', mu)
        
        y_obs = pm.Normal(f'y_obs_{i}', mu=mu, sigma=0.1, observed=concentration)

    # 使用 NUTS 采样算法进行 MCMC 采样
    trace = pm.sample(1000, tune=1000, cores=2)

# 绘制 MCMC 采样后的结果
pm.traceplot(trace)
plt.show()

# 打印后验分布的摘要
print(pm.summary(trace))

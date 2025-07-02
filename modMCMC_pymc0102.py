import numpy as np
import pandas as pd
from scipy.integrate import odeint
import pymc3 as pm
import theano.tensor as tt
from joblib import Parallel, delayed
from datetime import datetime
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
from scipy.stats import truncnorm, uniform

# 导入已有的 PBPK 建模函数
from pk_model1209 import log_normalize, exp_denormalize, derivshiv, pk_model


# 定义 MCMC 似然函数
def likelihood(log_params, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train, sig2):
    '''定义 MCMC 似然函数'''
    total_cost = 0
    for i in range(len(time_points_train)):
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        T_total = inject_timelen_train[i]
        D_total = dose
        timelen = time_points_train[i]
        y_pred = pk_model(timelen, D_total, T_total, *log_params)
        observed_values = concentration_data_train[i]
        cost = np.sum((concentration - y_pred) ** 2) / sig2
        total_cost += cost
    return -0.5 * total_cost

# 定义先验分布
def Prior(log_params):
    # Population level
    # 从一个参数向量中提取出与总体均值、标准差和误差方差相关的信息，
    # 然后将它们转换成适当的数值格式以供进一步的统计分析或建模使用。

    pars_data = np.exp(log_params)
    sig2 = 0.5  # 误差方差 error variances from model residual

    # 根据给定的参数向量，计算其均值，并根据预设的变异系数计算标准差。
    mean = np.exp(log_params)
    CV = 0.5  # Coefficient of variation; Default value of 0.5 in all parameters (Bois, 2000; Bois et al., 1996)
    sd = mean * CV  # 标准差

    # Calculate likelihoods of each parameters; P(u|M,S)
    a = np.array([truncnorm.ppf(0.025, (0 - m) / s, scale=s) for m, s in zip(mean, sd)])
    b = np.array([truncnorm.ppf(0.975, (0 - m) / s, scale=s) for m, s in zip(mean, sd)])

    prior_pars = truncnorm.pdf(pars_data, (a - mean) / sd, (b - mean) / sd, loc=mean, scale=sd)

    prior_sig2 = uniform.pdf(sig2, loc=0.01,
                             scale=3.3 - 0.01)  # error variances, Lower and upper boundary from Chiu et al., 2009; Chiu et al., 2014

    # log-transformed (log-likelihoods of each parameters)
    log_pri_pars = np.log(prior_pars)
    log_pri_sig2 = np.log(prior_sig2)

    # maximum likelihood estimation (MLE): negative log-likelihood function, (-2 times sum of log-likelihoods)
    MLE = -2 * (np.sum(log_pri_pars) + log_pri_sig2)

    return MLE


# 读取初始参数
from init_param import init_pars
sub_init_params = {
    "PRest": init_pars["PRest"],
    "PK": init_pars["PK"],
    "PL": init_pars["PL"],
    "Kbile": init_pars["Kbile"],
    "GFR": init_pars["GFR"],
    "Free": init_pars["Free"],
    "Vmax_baso": init_pars["Vmax_baso"],
    "Km_baso": init_pars["Km_baso"],
    "Kurine": init_pars["Kurine"],
    "Kreab": init_pars["Kreab"],
}
# 将初始参数转换为一个向量
sub_init_params_vec = np.array([sub_init_params[key] for key in sub_init_params])

#theta_MCMC = log_normalize(sub_init_params)
sig2 = 0.5


def modMCMC(time_points_train, concentration_data_train, input_dose_train, inject_timelen_train, niter, jump, burninlength):
    '''定义 MCMC 采样函数'''
    with pm.Model() as model:
        # 定义参数
        #theta_MCMC = pm.TruncatedNormal('theta_MCMC', mu=log_normalize(sub_init_params_vec), sigma=0.5, shape=len(sub_init_params_vec),
        #                                lower=np.minimum(0.5 * sub_init_params_vec, 0.01),
         #                               upper=np.maximum(2 * sub_init_params_vec, 100))

        sig2 = pm.Uniform('sig2', lower=0.01, upper=3.3)

        # 定义似然函数
        pm.Potential('likelihood', likelihood(sub_init_params_vec, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train, sig2))

        # 定义先验分布
        pm.Potential('prior', Prior(sub_init_params_vec))

        # 多链采样
        step = pm.Metropolis(scaling=jump)
        trace = pm.sample(niter, tune=burninlength, chains=4, cores=4, step=step)

        return trace



# 开始时间
start_time = datetime.now()
print(f"Start time: {start_time}")

# 并行运行 MCMC 采样
niter = 5000
jump = 0.01
burninlength = 1000
outputlength = 3000 

# 由于 pymc3 本身支持并行采样，这里直接使用 modMCMC 函数
trace = modMCMC(time_points_train, concentration_data_train, input_dose_train, inject_timelen_train, niter, jump,
                burninlength)#, outputlength

# 结束时间
end_time = datetime.now()
print(f"End time: {end_time}")
print(f"Total time: {end_time - start_time}")

# 收敛诊断
print(az.summary(trace))

# 保存结果
trace_df = pm.trace_to_dataframe(trace)
trace_df.to_csv("Human.summary_pos.csv", index=False)
trace_df.to_csv("Human.pos.csv", index=False)
az.to_netcdf(trace, "Human.comb.nc")

# 绘制迹线图
az.plot_trace(trace)
az.plot_posterior(trace)

# 显示图表
import matplotlib.pyplot as plt

plt.show()
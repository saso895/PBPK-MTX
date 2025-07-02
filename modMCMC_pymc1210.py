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

from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train

# warnings.filterwarnings('ignore')
theano.config.cxx = 'g++'

# 对数归一化和反归一化函数
def log_normalize(params):
    """对参数进行对数归一化"""
    return np.log(params)  # 确保对数变换只对正数操作
    # return np.log(np.array(params))
    # return np.log(np.array(list(params.values())))


def exp_denormalize(log_params):
    """对数归一化的反操作（指数恢复）"""
    return np.exp(log_params)
    # return np.exp(np.array(log_params))


# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, log_params):
    '''定义微分方程的函数，包含药物点滴输入'''
    # log_params = tt.reshape(log_params, (10,))
    # 反归一化参数（从对数空间恢复到原始空间）
    # parms = exp_denormalize(log_params)

    # PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = log_params
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
    input_rate = R if t <= T_total else 0

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


# 药代动力学模型函数，用于参数拟合
def pk_model(t, D_total, T_total, *normalized_params):
    '''药代动力学模型函数，用于参数拟合'''

    # # 对参数进行对数归一化
    # normalized_params = log_normalize(params)
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)

    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv,
        y0,
        t,
        args=(normalized_params, R, T_total),
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5  # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)
    return y[:, 0] / VPlas


# 自定义的成本函数：计算预测值和观测值之间的平方误差和
call_count = 0
def cost_function(params, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train):
    '''自定义的成本函数：计算预测值和观测值之间的平方误差和'''
    global call_count
    call_count += 1
    # print(f"Total cost 调用次数: {call_count}")
    total_cost = 0
    for i in range(len(time_points_train)):
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        T_total = inject_timelen_train[i]
        D_total = dose
        timelen = time_points_train[i]
        y_pred = pk_model(timelen, D_total, T_total, *params)
        # observed_values = concentration_data_train[i]
        cost = np.sum((concentration - y_pred) ** 2)
        # print(f"组 {idx + 1} 的成本: {cost}")
        total_cost += cost
    return total_cost


with open('pars/modfit_pars.pkl', 'rb') as f:
    average_params = pickle.load(f)

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


# 定义 ODE 函数时，接受额外的参数
# def ode_func(y, t, log_params):#, R, T_total
#     log_params = tt.reshape(log_params, (10,))
#     return ode_func_partial(y, t, log_params)#, R, T_total


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

    # 将参数包装成 Theano 的张量
    #log_params = tt.stack([PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab])
    #log_params = tt.as_tensor_variable([PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab])
    log_params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]
    # log_params = tt.stack([PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab])
    # log_params = tt.reshape(log_params, (10,))  # 显式指定形状为 (10,)
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
        ode_partial = pm.ode.DifferentialEquation(
            func=derivshiv,
            times=time,
            n_states=7,
            n_theta=10,
            t0=0
        )

        # 求解 ODE
        y = ode_partial(y0=tt.zeros(7), theta=log_params)

        #y = ode_partial(y0=tt.zeros(7), theta=log_params, extra_args={'R': R, 'T_total': T_total})

        # 计算预测浓度
        VPlas = tt.constant(VPlas)
        mu = y[:, 0] / VPlas
        
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
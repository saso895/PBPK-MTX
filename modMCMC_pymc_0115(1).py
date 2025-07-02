import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import theano.tensor as tt
import pymc3 as pm
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train

# warnings.filterwarnings('ignore')
# theano.config.cxx = 'g++'

# print('*** Start script ***')
# print(f'{pm.__name__}: v. {pm.__version__}')
# print(f'{theano.__name__}: v. {theano.__version__}')

random_seed=20394

# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, params):
    '''定义微分方程的函数，包含药物点滴输入'''
    # params = tt.reshape(params, (10,))
    # 反归一化参数（从对数空间恢复到原始空间）
    # parms = exp_denormalize(params)

    # PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = params
    PRest=params[0]
    PK=params[1]
    PL=params[2]
    Kbile=params[3]
    GFR=params[4]
    Free=params[5]
    Vmax_baso=params[6]
    Km_baso=params[7]
    Kurine = params[8]
    Kreab=params[9]

    # 确保 input_rate 是标量
    # input_rate = R if t <= T_total else 0
    input_rate = pm.math.switch(t > T_total, 0, R)
    

    # ydot = np.zeros(7)
    ydot0 = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (
                QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate
    ydot1 = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot2 = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (
                Km_baso + y[2] / VK / PK)
    ydot3 = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot4 = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[
        4] * Kurine - Kreab * y[4]
    ydot5 = Kurine * y[4]
    ydot6 = Kbile * y[1]

    return [ydot0,ydot1,ydot2,ydot3,ydot4,ydot5,ydot6]

if __name__ == '__main__':
    T_total = 0
    R = 0
    n_states=7
    n_theta=10

    pars=[]

    # 使用 pymc3 进行 MCMC 采样优化参数
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Normal('PRest', mu=init_pars["PRest"], sigma=1)
        PK = pm.Normal('PK', mu=init_pars["PK"], sigma=1)
        PL = pm.Normal('PL', mu=init_pars["PL"], sigma=1)
        Kbile = pm.Normal('Kbile', mu=init_pars["Kbile"], sigma=1)
        GFR = pm.Normal('GFR', mu=init_pars["GFR"], sigma=1)
        Free = pm.Normal('Free', mu=init_pars["Free"], sigma=0.1)
        Vmax_baso = pm.Normal('Vmax_baso', mu=init_pars["Vmax_baso"], sigma=10)
        Km_baso = pm.Normal('Km_baso', mu=init_pars["Km_baso"], sigma=1)
        Kurine = pm.Normal('Kurine', mu=init_pars["Kurine"], sigma=0.01)
        Kreab = pm.Normal('Kreab', mu=init_pars["Kreab"], sigma=0.01)

        # 将参数包装成 Theano 的张量
        params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]

        trace=None
        # params = tt.reshape(params, (10,))  # 显式指定形状为 (10,)
        # 遍历每组数据，逐组计算拟合值并添加观测数据的似然函数
        for i in range(len(time_points_train)):
            pars.append(params)
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
                n_states=n_states,
                n_theta=n_theta,
                t0=0
            )

            # 求解 ODE
            # y = ode_partial(y0=[0]*n_states, theta=params)

            #y = ode_partial(y0=tt.zeros(7), theta=params, extra_args={'R': R, 'T_total': T_total})

            # 计算预测浓度
            # VPlas = tt.constant(VPlas)
            # mu = y[:, 0] / VPlas
            
            # 使用 pm.Deterministic 来保存每一组预测值
            # pm.Deterministic(f'predicted_concentration_{i}', mu)

            # y_obs = pm.Normal(f'y_obs_{i}', mu=mu, sigma=0.1, observed=concentration, )


        # 使用 NUTS 采样算法进行 MCMC 采样
        trace = pm.sample(2000, chains=2, tune=1000, target_accept=0.9, cores=6, random_seed=random_seed)
        # data = az.from_pymc3(trace=trace)

        data = az.from_pymc3(trace=trace)
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
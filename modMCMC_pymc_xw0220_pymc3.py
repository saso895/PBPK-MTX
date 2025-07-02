import matplotlib.pyplot as plt
import theano.tensor as tt
# import theano
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import os
import pymc3 as pm
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from tqdm import tqdm
import pickle

random_seed=1

# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, params):
    '''定义微分方程的函数，包含药物点滴输入'''
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
    input_rate = pm.math.switch(t > T_total, 0, R)
    
    ydot0 = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate
    ydot1 = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot2 = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot3 = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot4 = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot5 = Kurine * y[4]
    ydot6 = Kbile * y[1]
    # print(ydot0)
    if isinstance(t, (int, float, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        print(f"t: {t:.2f}, PRest: {PRest.eval():.6f}, PK: {PK.eval():.6f}, PL: {PL.eval():.6f}, Kbile: {Kbile.eval():.6f}, GFR: {GFR.eval():.6f}, Free: {Free.eval():.6f}, Vmax_baso: {Vmax_baso.eval():.6f}, Km_baso: {Km_baso.eval():.6f}, Kurine: {Kurine.eval():.6f}, Kreab: {Kreab.eval():.6f}")
        print(f"t: {t:.2f}, ydot0: {ydot0.eval():.6f}, ydot1: {ydot1.eval():.6f}, ydot2: {ydot2.eval():.6f}, ydot3: {ydot3.eval():.6f}, ydot4: {ydot4.eval():.6f}, ydot5: {ydot5.eval():.6f}, ydot6: {ydot6.eval():.6f}")

    return [ydot0, ydot1, ydot2, ydot3, ydot4, ydot5, ydot6]#

n_states=7
n_theta=10

# 定义新的 ODE 求解器,,如果插值使用同一个时间网格，用这里
# ode_model = pm.ode.DifferentialEquation(
#                 func=derivshiv,
#                 times=time_points_train[0],
#                 n_states=n_states,
#                 n_theta=n_theta,
#                 t0=0
#             )

if __name__ == '__main__':
    T_total = 0
    R = 0    

    # 使用 pymc3 进行 MCMC 采样优化参数
    with pm.Model() as model:
        # 设置参数的先验分布
        P = np.array([init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"], \
                   init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"], init_pars["Kreab"]])
        params = pm.Normal('parameterGroup', mu=P, sigma=np.array([0.1]*10), shape=10)
        sigma = pm.HalfCauchy("sigma", 1)
        conc_data = pm.Data("conc_data", np.concatenate([d + 1e-5 for d in concentration_data_train]))
        # conc_data = pm.Data("conc_data", np.concatenate([d + 1e-5 for i, d in enumerate(concentration_data_train) if i<2]))

        y_obs = []
        for i in tqdm(range(len(time_points_train))):
        # for i in tqdm(range(2)):
            time_points = time_points_train[i]
            concentration = np.log(concentration_data_train[i]+10e-6)
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            D_total = dose
            T_total = timelen
            R = D_total / T_total

            # # 假设我们使用 pm.Data 来传递每个样本的数据
            # time_data = pm.Data("time_data", time_points_train)           # 列表或数组，每个元素为该样本时间点数组
            # conc_data = pm.Data("conc_data", np.concatenate([d + 1e-5 for d in concentration_data_train]))
            # dose_data = pm.Data("dose_data", input_dose_train)
            # Ttotal_data = pm.Data("Ttotal_data", inject_timelen_train)

            # 定义新的 ODE 求解器
            ode_partial = pm.ode.DifferentialEquation(
                func=derivshiv,
                times=time_points,
                n_states=n_states,
                n_theta=n_theta,
                t0=0
            )
            y0 =  tt.ones(7)
            # 求解 ODE
            y = ode_partial(y0=y0, theta=params)
            
            # 计算预测浓度
            
            y_obs.append(pm.Normal(f'y_obs_{i}', mu=y[:, 0] / VPlas, sigma=sigma, observed=concentration))
        #mu_all = tt.concatenate(mu_list)
            # 计算预测浓度
    
    #         y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration))
    #         #Y = pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration)
    #     step=pm.NUTS()
    #     trace = pm.sample(1000, step=step,chains=4, tune=500,  cores=20, discard_tuned_samples=False, random_seed=1)#target_accept=0.9,
    #     data = az.from_pymc3(trace=trace)
    #     summary = pm.summary(trace)
        
        #Y = pm.Normal(f'y_obs', mu=mu_all, sigma=sigma, observed=conc_data)
        
        trace = pm.sample(1000, chains=2, tune=1000, target_accept=0.9, cores=6, random_seed=random_seed, full_output=1)
        summary = pm.summary(trace)
        print(summary)
        params = summary.values[0:10]



        # 使用 NUTS 采样算法进行 MCMC 采样
        # trace = pm.sample(5000, chains=2, tune=2000, target_accept=0.9, cores=6, random_seed=random_seed)
        # data = az.from_pymc3(trace=trace)

        # data = az.from_pymc3(trace=trace)
        # # 绘制 MCMC 采样后的结果
        
        # pm.traceplot(trace)
        # plt.show()

        # 打印后验分布的摘要
        #print(pm.summary(trace))

        #summary = pm.summary(trace)
        
        #P = status.mean(axis=0)

        # 生成后验预测值
        # posterior_predictive = pm.sample_posterior_predictive(trace)

        # 从后验分布的摘要中提取均值
        best_params = [
            summary.loc['PRest', 'mean'],
            summary.loc['PK', 'mean'],
            summary.loc['PL', 'mean'],
            summary.loc['Kbile', 'mean'],
            summary.loc['GFR', 'mean'],
            summary.loc['Free', 'mean'],
            summary.loc['Vmax_baso', 'mean'],
            summary.loc['Km_baso', 'mean'],
            summary.loc['Kurine', 'mean'],
            summary.loc['Kreab', 'mean']
        ]
        np.save('saved_result/best_params.npy', best_params)

    

    # with tqdm(range(len(time_points_train))) as pbar:
    #     best_params = np.load('saved_result/best_params.npy')
    #     y_pred = []
    #     # 绘制观测值和后验预测值
    #     # pbar = tqdm(range(len(time_points_train)))
    #     for i in pbar:
    #     # for i in range(len(time_points_train)):
    #         pbar.set_description("Predicting sampe: ") # 设置描述
    #         time_points = time_points_train[i]
    #         concentration = concentration_data_train[i]
    #         dose = input_dose_train[i]
    #         timelen = inject_timelen_train[i]
    #         D_total = dose
    #         T_total = timelen
    #         R = D_total / T_total

    #         # 使用 partial 将 R 和 T_total 作为额外参数传递给 derivshiv
    #         # ode_func_partial = partial(derivshiv, R=R, T_total=T_total)

    #         # 定义新的 ODE 求解器
    #         ode_partial = pm.ode.DifferentialEquation(
    #             func=derivshiv,
    #             times=time_points,
    #             n_states=n_states,
    #             n_theta=n_theta,
    #             t0=0
    #         )

    #         # 求解 ODE
    #         y = ode_partial(y0=[0]*n_states, theta=best_params)
    #         # 计算预测浓度
    #         #VPlas = tt.constant(VPlas)
    #         predicted_concentration = y[:, 0] / VPlas   
    #         predicted_concentration = predicted_concentration.eval()         
    #         y_pred.append(predicted_concentration)


    # with open('saved_result/y_pred.pkl', 'wb') as f:
    #     pickle.dump(y_pred, f)

    # # with open('saved_result/y_pred.pkl', 'rb') as f:
	# #     y_pred = pickle.load(f)
    
    # #### --- 画图 --- ####
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # # 获取子图的行数和列数
    # num_groups = len(time_points_train)
    # rows = (num_groups + 2) // 3
    # cols = 3

    # # 创建画布，指定大小
    # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    # axes = axes.flatten()
    # for i in range(len(time_points_train)):
    #     time = time_points_train[i]
    #     concentration = concentration_data_train[i]
    #     dose = input_dose_train[i]
    #     timelen = inject_timelen_train[i]
    #     D_total = dose
    #     T_total = timelen
    #     pred_y=y_pred[i]
        
    #     # 在对应的子图上绘制散点和拟合曲线
    #     axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='red')
    #     axes[i].scatter(time, pred_y, label=f'拟合数据 组 {i+1}', color='green')
        
    #     axes[i].plot(time, pred_y, label=f'拟合曲线 组 {i+1}', color='blue')
    #     axes[i].set_xlabel('时间 (小时)')
    #     axes[i].set_ylabel('药物浓度 (ug/mL)')
    #     axes[i].set_title(f'药物浓度拟合 组 {i+1}')
    #     axes[i].legend()
        
    # # 如果子图数量不足，隐藏多余的子图
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # plt.tight_layout()

    # # 保存图像为矢量图格式
    # plt.savefig('saved_result/fit_results.svg', format='svg')
    # plt.show()


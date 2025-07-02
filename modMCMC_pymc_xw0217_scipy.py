import matplotlib.pyplot as plt
# import theano.tensor as tt
# import theano
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
# import theano.tensor as tt
import pymc3 as pm
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
# warnings.filterwarnings('ignore')
# theano.config.cxx = 'g++'

# print('*** Start script ***')
# print(f'{pm.__name__}: v. {pm.__version__}')
# print(f'{theano.__name__}: v. {theano.__version__}')

random_seed=20394

def derivshiv(y, t, parms, R, T_total):
    '''定义微分方程的函数，包含药物点滴输入'''
    
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    # 确保 input_rate 是标量
    input_rate = R if t <= T_total else 0

    ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate / VPlas
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]

    return ydot

# 药代动力学模型函数，用于参数拟合
def pk_model(t, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    # 计算注射速率
    R = D_total / T_total
    y0 = (10e-6)+np.zeros(7)
    
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        t, 
        args=(params, R, T_total), 
        # rtol=1e-4,  # 放宽相对误差容忍度
        # atol=1e-7,  # 放宽绝对误差容忍度
        # h0=1e-5     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)
    return y[:, 0] / VPlas

if __name__ == '__main__':

    # 使用 pymc3 进行 MCMC 采样优化参数
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Lognormal('PRest', mu=pm.math.log(init_pars["PRest"]), sigma=1)
        PK = pm.Lognormal('PK', mu=pm.math.log(init_pars["PK"]), sigma=1)
        PL = pm.Lognormal('PL', mu=pm.math.log(init_pars["PL"]), sigma=1)
        Kbile = pm.Lognormal('Kbile', mu=pm.math.log(init_pars["Kbile"]), sigma=1)
        GFR = pm.Lognormal('GFR', mu=pm.math.log(init_pars["GFR"]), sigma=1)
        Free = pm.Lognormal('Free', mu=pm.math.log(init_pars["Free"]), sigma=0.1)
        Vmax_baso = pm.Lognormal('Vmax_baso', mu=pm.math.log(init_pars["Vmax_baso"]), sigma=1)
        Km_baso = pm.Lognormal('Km_baso', mu=pm.math.log(init_pars["Km_baso"]), sigma=1)
        Kurine = pm.Lognormal('Kurine', mu=pm.math.log(init_pars["Kurine"]), sigma=0.01)
        Kreab = pm.Lognormal('Kreab', mu=pm.math.log(init_pars["Kreab"]), sigma=0.01)
        # 每一组预测值
        sigma = pm.HalfCauchy("sigma", 1)

        params = np.array([init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"], \
                   init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"], init_pars["Kreab"]])
       
        for i in tqdm(range(len(time_points_train))):
            # pars.append(params)
            time_points = time_points_train[i]
            concentration = concentration_data_train[i]+(10e-6)
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            D_total = dose
            T_total = timelen
            # R = D_total / T_total

            # 计算预测浓度
            mu = pk_model(time_points, D_total, T_total, *params) 

            Y = pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration)
            trace = pm.sample(2000, step=pm.Metropolis(),chains=2, tune=1000,  cores=28, discard_tuned_samples=False, random_seed=1)#target_accept=0.9,
            summary = pm.summary(trace)
            # print(summary)
            params = np.exp([
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
            ])
            # params[6] = np.log(params[6])

        # 打印后验分布的摘要
        # print(pm.summary(trace))

        summary = pm.summary(trace)
        print(summary)
        
        # pm.traceplot(trace)
        # plt.show()
        # 从后验分布的摘要中提取均值
        best_params = np.exp([
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
            ])
        # best_params[6] = np.log(params[6])
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


import matplotlib.pyplot as plt
import datetime
# import theano.tensor as tt
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
# import theano.tensor as tt
import pymc3 as pm
import arviz as az
from init_data import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train

import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd


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

# 药代动力学模型函数，用于浓度拟合
def pk_model(t, D_total, T_total, Duration,*param):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {param}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    # Specify time points to simulate
    Time=np.arange(0, Duration + 0.1, 0.1)
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        Time, 
        args=(param, R, T_total), 
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    #return y[:, 0] / VPlas
    CA = y[:, 0] / VPlas
    results = np.column_stack((Time, CA))
    return results

# 药代动力学模型函数，用于MCMC优化
def MC_model(t, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {params}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        t, 
        args=(params, R, T_total), 
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )        
    return y[:, 0] / VPlas


##############################--------原始参数+modfit参数+MCMC优化参数的拟合结果--------#################################################

# with open('saved_result/optimized_params.pkl', 'rb') as f:
#     fit_params = pickle.load(f)
#     #print("fit_params:", fit_params)


# ########-------使用 pymc3 进行 MCMC 采样优化参数--------########
# if __name__ == '__main__':
#     with pm.Model() as model:
#         # 设置参数的先验分布
#         PRest = pm.Lognormal('PRest', mu=pm.math.log(fit_params[0]), sigma=1)
#         PK = pm.Lognormal('PK', mu=pm.math.log(fit_params[1]), sigma=1)
#         PL = pm.Lognormal('PL', mu=pm.math.log(fit_params[2]), sigma=1)
#         Kbile = pm.Lognormal('Kbile', mu=pm.math.log(fit_params[3]), sigma=1)
#         GFR = pm.Lognormal('GFR', mu=pm.math.log(fit_params[4]), sigma=1)
#         Free = pm.Lognormal('Free', mu=pm.math.log(fit_params[5]), sigma=0.1)
#         Vmax_baso = pm.Lognormal('Vmax_baso', mu=pm.math.log(fit_params[6]), sigma=1)
#         Km_baso = pm.Lognormal('Km_baso', mu=pm.math.log(fit_params[7]), sigma=1)
#         Kurine = pm.Lognormal('Kurine', mu=pm.math.log(fit_params[8]), sigma=0.01)
#         Kreab = pm.Lognormal('Kreab', mu=pm.math.log(fit_params[9]), sigma=0.01)
#         # 每一组预测值
#         sigma = pm.HalfNormal("sigma", 1)

#         params = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], \
#                     fit_params[5], fit_params[6], fit_params[7], fit_params[8], fit_params[9]])
        
#                 # 创建一个列表来存储每个病人的观测变量
#         y_obs = []
        
#         for i in tqdm(range(len(time_points_train))):
#             # pars.append(params)
#             time_points = time_points_train[i]
#             concentration = concentration_data_train[i]+(10e-6)
#             dose = input_dose_train[i]
#             timelen = inject_timelen_train[i]
#             D_total = dose
#             T_total = timelen
            
#             # R = D_total / T_total

#             # 计算预测浓度
#             mu = MC_model(time_points, D_total, T_total, *params) 
#             #print(f"mu for y_obs_{i}: {mu}")  # 打印 mu 的值
#             y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration))
#             #Y = pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration)
#         step=pm.NUTS()
#         trace = pm.sample(1000, step=step,chains=4, tune=500,  cores=20, discard_tuned_samples=False, random_seed=1)#target_accept=0.9,
#         data = az.from_pymc3(trace=trace)
#         summary = pm.summary(trace)
#         print(pm.summary(trace))                                 
#         az.plot_trace(trace)
#         plt.show()    

#         best_params = [
#                 summary.loc['PRest', 'mean'],
#                 summary.loc['PK', 'mean'],
#                 summary.loc['PL', 'mean'],
#                 summary.loc['Kbile', 'mean'],
#                 summary.loc['GFR', 'mean'],
#                 summary.loc['Free', 'mean'],
#                 summary.loc['Vmax_baso', 'mean'],
#                 summary.loc['Km_baso', 'mean'],
#                 summary.loc['Kurine', 'mean'],
#                 summary.loc['Kreab', 'mean']
#             ]
#         # 获取当前日期
#         np.save('saved_result/best_params.npy', best_params)
        
#         # today_date = datetime.datetime.now().strftime('%Y-%m-%d')
#         # # 定义保存路径
#         # save_dir = f'saved_result/{today_date}'
#         # os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
#         # #np.save('save_dir', best_params)
#         # np.save(os.path.join(save_dir, 'best_params.npy'), best_params)
    
# ############---------MCMC优化之后的浓度预测------------#################
#     with tqdm(range(len(time_points_train))) as ybar:
#         best_params = np.load('saved_result/best_params.npy')
#         y_pred = []
#         # 绘制观测值和后验预测值
#         # pbar = tqdm(range(len(time_points_train)))
#         for i in ybar:
#         # for i in range(len(time_points_train)):
#             ybar.set_description("Predicting sampe: ") # 设置描述
#             time_points = time_points_train[i]
#             concentration = concentration_data_train[i]
#             dose = input_dose_train[i]
#             timelen = inject_timelen_train[i]
#             D_total = dose
#             T_total = timelen
#             Duration= time_points_train[i][-1]
#             #R = D_total / T_total

#             # 求解 ODE
#             y = pk_model(time_points, D_total, T_total, Duration,*best_params) 
#             # 计算预测浓度
#             #VPlas = tt.constant(VPlas)
#             predicted_concentration = y   
#             #predicted_concentration = predicted_concentration.eval()         
#             y_pred.append(predicted_concentration)

#             # 获取当前日期
#             # today_date = datetime.datetime.now().strftime('%Y-%m-%d')

#             # # 定义保存路径
#             # save_dir = f'saved_result/{today_date}'
#             # os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
#             # # 保存 y_pred
#             # file_path = os.path.join(save_dir, 'y_MCMC.pkl')
#             #file_path = f'saved_dir/y_pred.pkl'
#             with open('saved_result/y_MCMC.pkl', 'wb') as f:
#                 pickle.dump(y_pred, f)   


        
######### --- 画图 --- ########
with open('saved_result/y_MCMC0408.pkl', 'rb') as f:
    y_MCMC = pickle.load(f)
with open('saved_result/y_pred.pkl', 'rb') as f:
    y_pred = pickle.load(f)
with open('saved_result/y_fit.pkl', 'rb') as f:
    y_fit = pickle.load(f)
# with open('saved_result/y_MCMC0308.pkl', 'rb') as f:
#     y_MCMCmax = pickle.load(f)
# with open('saved_result/y_MCMC0309.pkl', 'rb') as f:
#     y_MCMCmax1 = pickle.load(f)

with tqdm(range(len(time_points_train))) as pbar:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取子图的行数和列数
    num_groups = len(time_points_train)
    rows = (num_groups + 2) // 3
    cols = 3

    # 创建画布，指定大小
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i in pbar:
        pbar.set_description("Predicting sampe: ") # 设置描述
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        Duration= time_points_train[i][-1]
        Time=np.arange(0, Duration + 0.1, 0.1)
        # dose = input_dose_train[i]
        # timelen = inject_timelen_train[i]
        # D_total = dose
        # T_total = timelen
        pred_y=y_pred[i]
        fit_y=y_fit[i]
        MCMC_y=y_MCMC[i]
        #MCMCmax_y=y_MCMCmax[i]
        #MCMCmax_y1=y_MCMCmax1[i]
        # 在对应的子图上绘制散点和拟合曲线
        axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='#E73235')              
        axes[i].plot(pred_y[:,0], pred_y[:,1], label=f'拟合曲线 组 {i+1}', color='#5ca788')
        axes[i].plot(fit_y[:,0], fit_y[:,1], label=f'fit曲线 组 {i+1}', color='#fe9015')
        axes[i].plot(MCMC_y[:,0], MCMC_y[:,1], label=f'MCMC曲线不同的sigma- 组 {i+1}', color='#227abc')
        #axes[i].plot(MCMCmax_y[:,0], MCMCmax_y[:,1], label=f'MCMCmax曲线 组 {i+1}', color='#da6598')
        #axes[i].plot(MCMCmax_y1[:,0], MCMCmax_y1[:,1], label=f'MCMCmax10曲线 组 {i+1}', color='#98a9d0')
        axes[i].set_xlabel('时间 (小时)')
        axes[i].set_ylabel('药物浓度 (ug/mL)')
        axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        axes[i].legend()
        # plt.show()
        
    # 如果子图数量不足，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # 保存图像为矢量图格式
    file_path = os.path.join('saved_result/', 'fit_results0421.svg')
    plt.savefig(file_path, format='svg')
    plt.show()

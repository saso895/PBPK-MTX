import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time,datetime
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

#==========读入模拟数据
with open('saved_result\GA_simu0506_0.pkl', 'rb') as f:
    y_GA=pickle.load( f)
with open('saved_result/SAData_0609_Powell.pkl', 'rb') as f:
    y_FIT=pickle.load( f)
with open('ini_params_simu0427.pkl', 'rb') as f:
    y_ini=pickle.load( f)
with open('mcmc_Data0612.pkl', 'rb') as f:
    y_fit_GA=pickle.load( f)    

####---SA结果   
with open('saved_result/MCMCData_0610_Powell.pkl','rb') as f:
    y_mcmc = pickle.load(f)
### --- 画图 --- ####
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
    all_predicted = []
    for i in pbar:
        pbar.set_description("Predicting sampe: ") # 设置描述
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        FIT_y=y_FIT[i]
        GA_y=y_GA[i]
        INI_y=y_ini[i]
        FITGA_y=y_fit_GA[i]
        MCMC_y=y_mcmc[i]

        #======搜集GOF数据=========#
        Time_full = FIT_y[:, 0]
        CA_full   = FIT_y[:, 1]
        # 从连续数据中提取对应时间点的浓度      
        predicted_concentration = np.interp(time, Time_full, CA_full)
        # 将该患者的预测值和观察值加入到列表中
        # 转换为 NumPy 数组，方便绘图
        #predicted_concentration = np.array(y_mcmc[:,1])
        all_predicted.extend(predicted_concentration)
        # 在对应的子图上绘制散点和拟合曲线
        axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='#E73235')              
        axes[i].plot(FIT_y[:,0], FIT_y[:,1], label=f'fit拟合曲线 组 {i+1}', color='#5ca788')
        #axes[i].plot(GA_y[:,0], GA_y[:,1], label=f'genetic曲线 组 {i+1}', color='#fdd363',lw=4)
        axes[i].plot(INI_y[:,0], INI_y[:,1], label=f'ini曲线 组 {i+1}', color='#227abc')
        #axes[i].plot(FITGA_y[:,0], FITGA_y[:,1], label=f'fitGA曲线 组 {i+1}', color='#b96d93')
        #axes[i].plot(MCMC_y[:,0], MCMC_y[:,1], label=f'MCMC曲线 组 {i+1}', color='#E73235')
        axes[i].set_xlabel('时间 (小时)')
        axes[i].set_ylabel('药物浓度 (mg/L)')
        axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        axes[i].legend()
        
    # 如果子图数量不足，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()

# 保存图像为矢量图格式
save_path =f'saved_result/Simuplot_0609.svg'
plt.savefig(save_path, format='svg')
plt.show()

################--------------绘制GOF图-----------------------
# 创建图形
plt.figure(figsize=(8, 8))

all_observed = np.concatenate(concentration_data_train) 

r2  = r2_score(all_observed, all_predicted)
rmse = np.sqrt(mean_squared_error(all_observed, all_predicted))

print(f"R² = {r2:.3f},  RMSE = {rmse:.2f}")
# 绘制散点图：x 为预测值，y 为观察值
plt.scatter(all_predicted, all_observed, color='blue', alpha=0.6, label='Prediction vs Observation')

# 绘制理想拟合线（y = x），表示完美拟合的情况
plt.plot([min(all_predicted), max(all_predicted)], 
         [min(all_predicted), max(all_predicted)], 
         color='red', linestyle='--', label='Ideal Fit')

# 设置图表标签和标题
plt.text(0.05, 0.95, f"$R^2 = {r2:.3f}$",
         transform=plt.gca().transAxes,
         fontsize=12, va='top')
plt.xlabel('Predicted Concentration')
plt.ylabel('Observed Concentration')
plt.title('Predicted vs Observed Concentrations (All Patients)')
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.savefig('GOF_{today_date}.png')
plt.show()
    

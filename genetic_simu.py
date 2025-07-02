import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import pymc as pm
print(pm.__file__)
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
#dosing_time_train,rate_data_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd

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
#TODO:传参顺序不对，把新版本的微分方程和model拷贝过来
def pk_model(t, D_total, T_total, Duration,*param):
    '''药代动力学模型函数，用于参数拟合'''
    print(f"params : {param}") 
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
    CA = y[:, 0] / VPlas
    results = np.column_stack((t, CA))
    return results



##############################--------原始参数 + modfit参数的拟合结果--------#################################################
# 获取当前日期
#today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
with open('saved_result/optimized_params0425.pkl', 'rb') as f:
    fit_params = pickle.load(f)
    #print("fit_params:", fit_params)
    params = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], \
                fit_params[5], fit_params[6], fit_params[7], fit_params[8], fit_params[9]])
    init_params = [init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"],
                 init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"],
             init_pars["Kreab"]]
    # 优化后的参数值
    genetic_params = [
        0.020000000000000004, 
        1.6670765781402583, 
        0.466, 
        1.4526947593688966, 
        1.3901312351226809, 
        0.00040609884262084964, 
        16.66244625735275, 
        178.14, 
        0.006300594806671143, 
        1.0
    ]
            # 创建一个列表来存储每个病人的观测变量
    y_mu = []
    #y_obs = []
    for i in tqdm(range(len(time_points_train))):
        # pars.append(params)
        Duration= time_points_train[i][-1]
        time_points = time_points_train[i]
        concentration = concentration_data_train[i]+(10e-6)
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen
        
        # 计算预测浓度
        mu = pk_model(time_points, D_total, T_total,Duration, *genetic_params)
        y_mu.append(mu)  
with open('genetic_params_simu0425.pkl', 'wb') as f:
    pickle.dump(y_mu, f)

print("modfit拟合结果已保存到 fit_simu0425.pkl")

### --- 画图 --- ####

###with open('D:\BaiduSyncdisk\MTX\MTXmodel\saved_result\y_pred.pkl', 'wb') as f:
 #   y_pred = pickle.load(f)

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
        #pred_y=y_pred[i]
        mu_y=y_mu[i]
        # 在对应的子图上绘制散点和拟合曲线
        axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='red')              
        #axes[i].plot(Time, pred_y, label=f'拟合曲线 组 {i+1}', color='blue')
        axes[i].plot(Time, mu_y, label=f'genetic曲线 组 {i+1}', color='green')
        axes[i].set_xlabel('时间 (小时)')
        axes[i].set_ylabel('药物浓度 (ug/mL)')
        axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        axes[i].legend()
        
    # 如果子图数量不足，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()

# 保存图像为矢量图格式
file_path = os.path.join('fit_results0425.svg')
plt.savefig(file_path, format='svg')
plt.show()

################绘制GOF图
def GOF_model(time_points, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    R = D_total / T_total
    y0 = np.zeros(7)
    
    y = odeint(
        derivshiv, 
        y0, 
        time_points, 
        args=(params, R, T_total), 
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )
    return y[:, 0] / VPlas
# 创建图形
plt.figure(figsize=(8, 8))

# 初始化存储所有患者的预测值和观察值的列表
all_predicted = []
all_observed = []

genetic_params = [
        0.020000000000000004, 
        1.6670765781402583, 
        0.466, 
        1.4526947593688966, 
        1.3901312351226809, 
        0.00040609884262084964, 
        16.66244625735275, 
        178.14, 
        0.006300594806671143, 
        1.0
    ]

#param_means_norm = normalize_params(param_means, lower_bounds, upper_bounds)
    # 遍历每个患者的数据
for i in range(len(time_points_train)):
    time = time_points_train[i]
    concentration = concentration_data_train[i]
    dose = input_dose_train[i]
    timelen = inject_timelen_train[i]
    
    # 计算预测值 (使用参数均值)
    predicted_concentration = GOF_model(time, D_total, T_total,Duration, *genetic_params)  # 使用均值参数进行预测

    # 将该患者的预测值和观察值加入到列表中
    all_predicted.extend(predicted_concentration)
    all_observed.extend(concentration)

# 转换为 NumPy 数组，方便绘图
all_predicted = np.array(all_predicted)
all_observed = np.array(all_observed)

# 绘制散点图：x 为预测值，y 为观察值
plt.scatter(all_predicted, all_observed, color='blue', alpha=0.6, label='Prediction vs Observation')

# 绘制理想拟合线（y = x），表示完美拟合的情况
plt.plot([min(all_predicted), max(all_predicted)], 
         [min(all_predicted), max(all_predicted)], 
         color='red', linestyle='--', label='Ideal Fit')

# 设置图表标签和标题
plt.xlabel('Predicted Concentration')
plt.ylabel('Observed Concentration')
plt.title('Predicted vs Observed Concentrations (All Patients)')
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.savefig('predicted_vs_observed_all_patients0425.png')
plt.show()
    

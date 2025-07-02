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
#import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd

#random_seed=20394

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

    CA = y[:, 0] / VPlas
    results = np.column_stack((Time, CA))
    return results

# 药代动力学模型函数，用于MCMC优化
def FIT_model(t, D_total, T_total, *params):
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
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    return y[:, 0] / VPlas

##############################--------原始参数+modfit参数+MCMC优化参数的拟合结果--------#################################################
# 获取当前日期
#today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
with open('saved_result/optimized_params.pkl', 'rb') as f:
    fit_params = pickle.load(f)
    #print("fit_params:", fit_params)
if __name__ == '__main__':

    ########-------使用 pymc3 进行 MCMC 采样优化参数--------########
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Lognormal('PRest', mu=pm.math.log(fit_params[0]), sigma=0.5)
        PK = pm.Lognormal('PK', mu=pm.math.log(fit_params[1]), sigma=0.5)
        PL = pm.Lognormal('PL', mu=pm.math.log(fit_params[2]), sigma=0.5)
        Kbile = pm.Lognormal('Kbile', mu=pm.math.log(fit_params[3]), sigma=0.5)
        GFR = pm.Lognormal('GFR', mu=pm.math.log(fit_params[4]), sigma=0.5)
        Free = pm.Lognormal('Free', mu=pm.math.log(fit_params[5]), sigma=0.2)
        Vmax_baso = pm.Lognormal('Vmax_baso', mu=pm.math.log(fit_params[6]), sigma=3)
        Km_baso = pm.Lognormal('Km_baso', mu=pm.math.log(fit_params[7]), sigma=1)
        Kurine = pm.Lognormal('Kurine', mu=pm.math.log(fit_params[8]), sigma=0.1)
        Kreab = pm.Lognormal('Kreab', mu=pm.math.log(fit_params[9]), sigma=0.1)
        # 每一组预测值
        sigma = pm.HalfNormal("sigma", 1)

        params = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], \
                   fit_params[5], fit_params[6], fit_params[7], fit_params[8], fit_params[9]])
       
                # 创建一个列表来存储每个病人的观测变量
        y_obs = []
        
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
            mu = FIT_model(time_points, D_total, T_total, *params) 
            #weights = np.linspace(1.0, 0.5, len(time_points))
                # ====== 新增权重设置 ======
            # 创建布尔掩码
            is_zero = (time_points == 0)
            is_0_24 = (time_points > 0) & (time_points < 24)
            is_24_plus = (time_points >= 24)

            # 三阶段权重分配
            weights = np.select(
                condlist=[is_zero, is_0_24, is_24_plus],
                choicelist=[0.0, 1.0, 0.1],  # 0小时/0-24/24+
                default=0.1  # 兜底值
            )
            
            # 处理0权重避免除零错误
            weights = np.where(weights == 0, 1e-6, weights)
            
            weighted_sigma = sigma / weights  # 最终标准差
            # ====== 权重设置结束 ======
            
            #print(f"mu for y_obs_{i}: {mu}")  # 打印 mu 的值
            y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=weighted_sigma, observed=concentration))
            
        # 使用 MAP 获得初始值
        start = pm.find_MAP(method="BFGS")
        step=pm.NUTS()
        trace = pm.sample(1000, step=step,start=start,chains=10, tune=200,  cores=20, discard_tuned_samples=True, random_seed=1)#target_accept=0.9,
        data = az.from_pymc3(trace=trace)
        summary = pm.summary(trace)
        print(pm.summary(trace))                                 
        az.plot_trace(trace)
        plt.show()    

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
        # 获取当前日期
        #today_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # 定义保存路径
        # save_dir = f'saved_result/{today_date}'
        # os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
        #np.save('save_dir', best_params)
        np.save(os.path.join('saved_result/', 'best_params0421.npy'), best_params)
     
    ############---------MCMC优化之后的浓度预测------------#################
    with tqdm(range(len(time_points_train))) as ybar:
        best_params = np.load('saved_result/best_params0421.npy')
        y_pred = []
        # 绘制观测值和后验预测值
        # pbar = tqdm(range(len(time_points_train)))
        for i in ybar:
        # for i in range(len(time_points_train)):
            ybar.set_description("Predicting sampe: ") # 设置描述
            time_points = time_points_train[i]
            concentration = concentration_data_train[i]
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            D_total = dose
            T_total = timelen
            Duration= time_points_train[i][-1]
            #R = D_total / T_total

            # 求解 ODE
            y = pk_model(time_points, D_total, T_total, Duration,*best_params) 
            # 计算预测浓度
            #VPlas = tt.constant(VPlas)
            predicted_concentration = y   
            #predicted_concentration = predicted_concentration.eval()         
            y_pred.append(predicted_concentration)

            # 获取当前日期
            #today_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # 定义保存路径
            # save_dir = f'saved_result/{today_date}'
            # os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
            # 保存 y_pred
    save_dir = 'saved_result'
    file_path = os.path.join(save_dir, 'y_MCMC0408.pkl')
        #file_path = os.path.join('saved_result/', 'y_MCMC0305.pkl')
            #file_path = f'saved_dir/y_pred.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(y_pred, f)   


        











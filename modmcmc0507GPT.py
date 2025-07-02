import matplotlib.pyplot as plt
import datetime
from theano.compile.ops import as_op
import theano
import theano.tensor as tt
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import pymc as pm
from pymc.ode import DifferentialEquation
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
import aesara.tensor as at

os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
#random_seed=20394

def derivshiv(y, t, parms, R, T_total):
    '''定义微分方程的函数，包含药物点滴输入'''
    
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    # 确保 input_rate 是标量
    input_rate = at.switch(at.le(t, T_total), R, 0)
    #input_rate = R if t <= T_total else 0
    ydot = at.zeros(7) 
    #ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate / VPlas
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]
    return ydot


# 药代动力学模型函数，用于MCMC优化

def mcmc_model(t, D_total, T_total, PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab):
#def FIT_model(t, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {params}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    params = [
        PRest, PK, PL, Kbile, GFR, 
        Free, Vmax_baso, Km_baso, Kurine, Kreab
    ]
    #times_np = np.asarray(t, dtype=float)
    def deriv_fixed(y, tt, theta):
        return derivshiv(y, tt, theta, R, T_total)
    #params_values = [param.eval() if isinstance(param, pm.model.FreeRV) else param for param in params]
    ODE_SYSTEM = pm.ode.DifferentialEquation(
        func=deriv_fixed,  # 使用原来的微分方程
        times=t,  # 使用实际时间点，而非固定的 np.linspace(0, 24, 121)
        n_states=7,  # 微分方程的状态变量数目
        n_theta=10,  # 需要拟合的参数数目
        t0=0.0#,  # 初始时间
        #args=(R,) 
    )
    y = ODE_SYSTEM(y0=y0, theta=params, times=t)
    CA = y[:, 0] / VPlas
    results = np.column_stack((t, CA))
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    return results

##############################--------原始参数+modfit参数+MCMC优化参数的拟合结果--------#################################################
# 获取当前日期
#today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
with open('saved_result/optimized_params0427_Powell.pkl', 'rb') as f:
    fit_params = pickle.load(f)
    #print("fit_params:", fit_params)
if __name__ == '__main__':

    ########-------使用 pymc3 进行 MCMC 采样优化参数--------########
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Lognormal('PRest', mu=pm.math.log(fit_params[0]), sigma=0.3)
        PK = pm.Lognormal('PK', mu=pm.math.log(fit_params[1]), sigma=0.2)
        PL = pm.Lognormal('PL', mu=pm.math.log(fit_params[2]), sigma=0.2)
        Kbile = pm.Lognormal('Kbile', mu=pm.math.log(fit_params[3]), sigma=0.2)
        GFR = pm.Lognormal('GFR', mu=pm.math.log(fit_params[4]), sigma=0.1)
        Free = pm.Beta('Free', alpha=2, beta=2)#pm.Lognormal('Free', mu=pm.math.log(fit_params[5]), sigma=0.1)
        Vmax_baso = pm.Lognormal('Vmax_baso', mu=pm.math.log(fit_params[6]), sigma=0.5)
        Km_baso = pm.Lognormal('Km_baso', mu=pm.math.log(fit_params[7]), sigma=0.3)
        Kurine = pm.Lognormal('Kurine', mu=pm.math.log(fit_params[8]), sigma=0.2)
        Kreab = pm.Lognormal('Kreab', mu=pm.math.log(fit_params[9]), sigma=0.2)
        # 每一组预测值
        sigma = pm.HalfNormal("sigma", 1)
        #params_model = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]
        #params = np.array([fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], \
        #           fit_params[5], fit_params[6], fit_params[7], fit_params[8], fit_params[9]])
        
                # 创建一个列表来存储每个病人的观测变量
        y_obs = []
        
        for i in tqdm(range(len(time_points_train))):
            # pars.append(params)
            time_points = time_points_train[i].astype(np.float64)
            D_total = tt.as_tensor_variable(input_dose_train[i].astype(np.float64))
            T_total = tt.as_tensor_variable(inject_timelen_train[i].astype(np.float64))
        
            #time_points = time_points_train[i]
            concentration = concentration_data_train[i]#+(10e-6)
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            #D_total = dose
            #T_total = timelen        
            
            # 计算预测浓度
            mu = mcmc_model(
            time_points, D_total, T_total, 
            PRest, PK, PL, Kbile, GFR, Free, 
            Vmax_baso, Km_baso, Kurine, Kreab
            ) 
            sigma = pm.HalfNormal(f'sigma_{i}', 1)  # 每个病人的浓度观测都有自己的误差
            pm.Normal(f'y_obs_{i}', mu=mu[:, 0], sigma=sigma, observed=concentration)

            #y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu[:, 0], sigma=sigma, observed=concentration))
        # 使用 MAP 获得初始值
        start = pm.find_MAP(method="BFGS")
        #step=pm.Metropolis()
        trace = pm.sample(draws=1000, 
                          tune=500,
                          init='adapt_diag', 
                          #step=step,
                          chains=2,  
                          cores=4, 
                          discard_tuned_samples=True, 
                          random_seed=1)#target_accept=0.9,start=start,
        #data = az.from_pymc3(trace=trace)
        summary = pm.summary(trace)
        print(pm.summary(trace))       
        #ess_bulk = az.ess(trace, method="bulk")
        #ess_tail = az.ess(trace, method="tail")

        #print("Effective Sample Size (ESS) Bulk:")
        #print(ess_bulk)

        #print("Effective Sample Size (ESS) Tail:")
        #print(ess_tail)                          
        #az.plot_trace(trace)
        #plt.savefig('trace11.svg')
        #az.plot_posterior(trace, var_names=['GFR', 'Free', 'PRest', 'PK', 'PL', 'Kbile', 'Vmax_baso', 'Km_baso', 'Kurine', 'Kreab'])
        #plt.savefig('posterio11.svg')
        #az.plot_autocorr(trace, var_names=['GFR', 'Free', 'PRest', 'PK', 'PL', 'Kbile', 'Vmax_baso', 'Km_baso', 'Kurine', 'Kreab'])
        #plt.savefig('autocorr11.svg')
        #plt.show()    

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
        #np.save(os.path.join('saved_result/', 'best_params0428.npy'), best_params)
        with open('saved_result/best_params0507_GPT.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        print("MCMC优化参数已保存")
     
    ############---------MCMC优化之后的浓度预测------------#################
    with tqdm(range(len(time_points_train))) as ybar:
        with open('saved_result/best_params0507.pkl', 'rb') as f:
            best_params = pickle.load(f)
        #best_params = np.load('saved_result/best_params0428.npy')
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

    with open('saved_result\y_MCMC0507.pkl', 'wb') as f:
        pickle.dump(y_pred, f)   
    print("MCMC拟合数据已保存")


        











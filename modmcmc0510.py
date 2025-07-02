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
import pymc3 as pm
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
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
        method='BDF',
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )

    CA = y[:, 0] / VPlas
    results = np.column_stack((Time, CA))
    return results

# 药代动力学模型函数，用于MCMC优化
@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar]+ [tt.dscalar for _ in range(10)], otypes=[tt.dvector])
def theano_FIT_model(t, D_total, T_total, PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab):
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
    #params_values = [param.eval() if isinstance(param, pm.model.FreeRV) else param for param in params]
 
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        t, 
        args=(params , R, T_total), 
        rtol=1e-3,  # 放宽相对误差容忍度
        atol=1e-5,  # 放宽绝对误差容忍度
        mxstep=5000
        #h0=0.1     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    return y[:, 0] / VPlas

##############################--------原始参数+modfit参数+MCMC优化参数的拟合结果--------#################################################
# 获取当前日期
#today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# 定义保存路径
#save_dir = f'saved_result/{today_date}'
#os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
with open('/nfs/home/y18300744/MTXmodel/saved_result/optimized_params0427_Powell.pkl', 'rb') as f:
    fit_params = pickle.load(f)
    #print("fit_params:", fit_params)
if __name__ == '__main__':

    ########-------使用 pymc3 进行 MCMC 采样优化参数--------########
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Lognormal('PRest', mu=pm.math.log(fit_params[0]), sigma=3)
        PK = pm.Lognormal('PK', mu=pm.math.log(fit_params[1]), sigma=0.2)
        PL = pm.Lognormal('PL', mu=pm.math.log(fit_params[2]), sigma=0.5)
        Kbile = pm.Lognormal('Kbile', mu=pm.math.log(fit_params[3]), sigma=0.5)
        GFR = pm.HalfNormal('GFR', sigma=50)
        #GFR = pm.Lognormal('GFR', mu=pm.math.log(fit_params[4]), sigma=3)
        Free = pm.Beta('Free', alpha=2, beta=2)#pm.Lognormal('Free', mu=pm.math.log(fit_params[5]), sigma=0.1)
        Vmax_baso = pm.Lognormal('Vmax_baso', mu=pm.math.log(fit_params[6]), sigma=1)
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
            time_points = tt.as_tensor_variable(time_points_train[i].astype(np.float64))
            D_total = tt.as_tensor_variable(input_dose_train[i].astype(np.float64))
            T_total = tt.as_tensor_variable(inject_timelen_train[i].astype(np.float64))
        
            #time_points = time_points_train[i]
            concentration = concentration_data_train[i]#+(10e-6)
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            #D_total = dose
            #T_total = timelen        
            
            # 计算预测浓度
            mu = theano_FIT_model(
            time_points, D_total, T_total, 
            PRest, PK, PL, Kbile, GFR, Free, 
            Vmax_baso, Km_baso, Kurine, Kreab
            )
            #mu = FIT_model(time_points, D_total, T_total, PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab) 
            #weights = np.linspace(1.0, 0.5, len(time_points))
                # ====== 新增权重设置 ======
            # 创建布尔掩码
            # is_zero = (time_points == 0)
            # is_0_24 = (time_points > 0) & (time_points < 24)
            # is_24_plus = (time_points >= 24)

            # # 三阶段权重分配
            # weights = np.select(
            #     condlist=[is_zero, is_0_24, is_24_plus],
            #     choicelist=[0.0, 1.0, 0.1],  # 0小时/0-24/24+
            #     default=0.1  # 兜底值
            # )
            
            # # 处理0权重避免除零错误
            # weights = np.where(weights == 0, 1e-6, weights)
            
            # weighted_sigma = sigma / weights  # 最终标准差
            # ====== 权重设置结束 ======
            
            #print(f"mu for y_obs_{i}: {mu}")  # 打印 mu 的值
            #y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=weighted_sigma, observed=concentration))
            y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration))
        # 使用 MAP 获得初始值
        start = pm.find_MAP(method="BFGS")
        step = pm.DEMetropolisZ(blocked=True) 
        #step=pm.Metropolis()
        trace = pm.sample(draws=4000, 
                          tune=2000,
                          #init='adapt_diag', 
                          start=start,
                          step=step,
                          chains=4,  
                          cores=4, 
                          discard_tuned_samples=True, 
                          return_inferencedata=False,
                          progressbar=True,
                          random_seed=1)#target_accept=0.9,
        #data = az.from_pymc3(trace=trace)
        summary = pm.summary(trace)
        print(pm.summary(trace))       
        #ess_bulk = az.ess(trace, method="bulk")
        #ess_tail = az.ess(trace, method="tail")

        #print("Effective Sample Size (ESS) Bulk:")
        #print(ess_bulk)

        #print("Effective Sample Size (ESS) Tail:")
        #print(ess_tail)                          
        az.plot_trace(trace)
        plt.savefig('trace0509.svg')
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
        with open('saved_result/best_params0509.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        print("MCMC优化参数已保存")
     
    

        











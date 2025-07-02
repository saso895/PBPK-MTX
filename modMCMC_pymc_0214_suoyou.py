import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import numpy as np
from functools import partial
from init_param import init_pars, QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train

random_seed = 20394

# 生成数据集（按患者分组）
def generate_dataset():
    patient_data = []
    for i in range(len(time_points_train)):
        patient = {
            'times': time_points_train[i],
            'concentration': concentration_data_train[i],
            'R': input_dose_train[i]/inject_timelen_train[i],
            'T_total': inject_timelen_train[i]
        }
        patient_data.append(patient)
    return patient_data

patient_data = generate_dataset()

# 修正的ODE函数
def derivshiv(y, t, params, R_patients, T_total_patients, patient_idx):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = params
    R = R_patients[patient_idx]
    T_total = T_total_patients[patient_idx]
    
    input_rate = pm.math.switch(t > T_total, 0.0, R)
    
    ydot0 = (QRest*y[3]/VRest/PRest + QK*y[2]/VK/PK + QL*y[1]/VL/PL - 
             QPlas*y[0]/VPlas + Kreab*y[4] + input_rate)
    ydot1 = QL*(y[0]/VPlas - y[1]/VL/PL) - Kbile*y[1]
    ydot2 = QK*(y[0]/VPlas - y[2]/VK/PK) - y[0]/VPlas*GFR*Free - \
            (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK)
    ydot3 = QRest*(y[0]/VPlas - y[3]/VRest/PRest)
    ydot4 = y[0]/VPlas*GFR*Free + (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK) - \
            y[4]*Kurine - Kreab*y[4]
    ydot5 = Kurine*y[4]
    ydot6 = Kbile*y[1]
    
    return [ydot0, ydot1, ydot2, ydot3, ydot4, ydot5, ydot6]

n_states = 7
n_params = 10

with pm.Model() as model:
    # 参数先验（示例调整）
    PRest = pm.Normal('PRest', mu=np.log(init_pars["PRest"]), sigma=0.1)
    PK = pm.Normal('PK', mu=np.log(init_pars["PK"]), sigma=0.1)
    PL = pm.Normal('PL', mu=np.log(init_pars["PL"]), sigma=0.1)
    Kbile = pm.Normal('Kbile', sigma=1)
    GFR = pm.Normal('GFR', mu=init_pars["GFR"], sigma=5)
    Free = pm.Beta('Free', alpha=2, beta=2)  # 假设Free是比例参数
    Vmax_baso = pm.Normal('Vmax_baso', mu=np.log(init_pars["Vmax_baso"]), sigma=0.5)
    Km_baso = pm.Normal('Km_baso', mu=np.log(init_pars["Km_baso"]), sigma=0.5)
    Kurine = pm.Normal('Kurine', sigma=0.1)
    Kreab = pm.Normal('Kreab', sigma=0.1)
    
    params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]
    
    # 患者特定参数
    R_vals = [p['R'] for p in patient_data]
    T_total_vals = [p['T_total'] for p in patient_data]
    R_shared = pm.Data('R_shared', R_vals)
    T_total_shared = pm.Data('T_total_shared', T_total_vals)
      
    # 分患者求解ODE 
    y_preds = []
    for i, pat in enumerate(patient_data):
        ode_i = pm.ode.DifferentialEquation(
            func=partial(derivshiv, R_patients=R_shared, 
                        T_total_patients=T_total_shared, patient_idx=i),
            times=pat['times'],
            n_states=n_states,
            n_theta=n_params
        )
        y_pred = ode_i(y0=[0.0]*n_states, theta=params)
        y_preds.append(y_pred[:,0]/VPlas)  # 预测血浆浓度
    
    # 合并所有患者的预测
    y_combined = pm.math.concatenate(y_preds)
    
    # 似然函数
    obs_combined = np.concatenate([p['concentration'] for p in patient_data])
    Y_obs = pm.Normal('Y_obs', mu=y_combined, sd=0.1, observed=obs_combined)
    
    # 采样
    trace = pm.sample(1000, tune=1000, target_accept=0.95, cores=4, 
                     random_seed=random_seed)
    
    # 后验预测检查
    ppc = pm.sample_posterior_predictive(trace, var_names=['Y_obs'])
    
    # 可视化
    az.plot_trace(trace)
    plt.show()
    
    # 分患者绘图
    start_idx = 0
    for i, pat in enumerate(patient_data):
        end_idx = start_idx + len(pat['times'])
        plt.figure(figsize=(10,4))
        plt.plot(pat['times'], obs_combined[start_idx:end_idx], 'ko', label='Observed')
        plt.plot(pat['times'], ppc['Y_obs'][:, start_idx:end_idx].mean(axis=0), 
                'r-', label='Predicted Mean')
        plt.fill_between(pat['times'],
                        ppc['Y_obs'][:, start_idx:end_idx].mean(0) - 2*ppc['Y_obs'][:, start_idx:end_idx].std(0),
                        ppc['Y_obs'][:, start_idx:end_idx].mean(0) + 2*ppc['Y_obs'][:, start_idx:end_idx].std(0),
                        alpha=0.3, color='red')
        plt.title(f'Patient {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()
        start_idx = end_idx
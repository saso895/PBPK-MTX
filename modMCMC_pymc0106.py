import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.integrate import odeint
from init_param import PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
from pk_model1209 import pk_model, cost_function, log_normalize, exp_denormalize

# 定义优化参数的名称和初始值
opt_params_names = ["PRest", "PK", "PL", "Kbile", "GFR", "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab"]
opt_params_init = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]

# 对初始参数进行对数归一化
log_opt_params_init = log_normalize(opt_params_init)

# 定义贝叶斯模型
with pm.Model() as model:
    # 定义先验分布
    PRest = pm.Normal('PRest', mu=log_opt_params_init[0], sigma=1)
    PK = pm.Normal('PK', mu=log_opt_params_init[1], sigma=1)
    PL = pm.Normal('PL', mu=log_opt_params_init[2], sigma=1)
    Kbile = pm.Normal('Kbile', mu=log_opt_params_init[3], sigma=1)
    GFR = pm.Normal('GFR', mu=log_opt_params_init[4], sigma=1)
    Free = pm.Normal('Free', mu=log_opt_params_init[5], sigma=1)
    Vmax_baso = pm.Normal('Vmax_baso', mu=log_opt_params_init[6], sigma=1)
    Km_baso = pm.Normal('Km_baso', mu=log_opt_params_init[7], sigma=1)
    Kurine = pm.Normal('Kurine', mu=log_opt_params_init[8], sigma=1)
    Kreab = pm.Normal('Kreab', mu=log_opt_params_init[9], sigma=1)

    # 反归一化参数
    PRest_denorm = pm.Deterministic('PRest_denorm', pm.math.exp(PRest))
    PK_denorm = pm.Deterministic('PK_denorm', pm.math.exp(PK))
    PL_denorm = pm.Deterministic('PL_denorm', pm.math.exp(PL))
    Kbile_denorm = pm.Deterministic('Kbile_denorm', pm.math.exp(Kbile))
    GFR_denorm = pm.Deterministic('GFR_denorm', pm.math.exp(GFR))
    Free_denorm = pm.Deterministic('Free_denorm', pm.math.exp(Free))
    Vmax_baso_denorm = pm.Deterministic('Vmax_baso_denorm', pm.math.exp(Vmax_baso))
    Km_baso_denorm = pm.Deterministic('Km_baso_denorm', pm.math.exp(Km_baso))
    Kurine_denorm = pm.Deterministic('Kurine_denorm', pm.math.exp(Kurine))
    Kreab_denorm = pm.Deterministic('Kreab_denorm', pm.math.exp(Kreab))

    # 定义观测数据
    observed_concentrations = concentration_data_train

    # 定义似然函数
    for i in range(len(time_points_train)):
        observed_data = observed_concentrations[i]
        time_points = time_points_train[i]
        dose = input_dose_train[i]
        T_total = inject_timelen_train[i]

        # 定义模型预测
        predicted_concentrations = pm.Deterministic(f'predicted_concentrations_{i}', pk_model(time_points, dose, T_total, PRest_denorm, PK_denorm, PL_denorm, Kbile_denorm, GFR_denorm, Free_denorm, Vmax_baso_denorm, Km_baso_denorm, Kurine_denorm, Kreab_denorm))

        # 定义似然函数
        pm.Normal(f'likelihood_{i}', mu=predicted_concentrations, sigma=0.1, observed=observed_data)

    # 使用MCMC方法进行采样
    with model:
        trace = pm.sample(draws=20000, chains=4, cores=None, tune=5000, random_seed=42)

# 保存采样结果
pm.save_trace(trace, directory='mcmc_trace', overwrite=True)

# 打印优化后的参数
for param_name in opt_params_names:
    print(f"{param_name}: {np.mean(exp_denormalize(trace[param_name]))}")

# 保存优化后的参数
optimized_params = {param_name: np.mean(exp_denormalize(trace[param_name])) for param_name in opt_params_names}
optimized_params_df = pd.DataFrame([optimized_params])
optimized_params_df.to_csv('optimized_params.csv', index=False)
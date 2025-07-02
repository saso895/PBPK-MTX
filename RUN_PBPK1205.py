import numpy as np
from scipy.integrate import odeint
import pandas as pd
from pars1111 import params
from init_data1029 import data, arm_data_train,time_points_train, time_points_test, concentration_data_train, concentration_data_test, rate_data_train, rate_data_test, dosing_time_train, dosing_time_test
from MTX_PBPK1205 import MTX_PBPK, total_cost, log_normalize, exp_denormalize, prior_log_dist,log_likelihood
import time
import warnings
import matplotlib.pyplot as plt
import pymc as pm
import pickle
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import emcee
import pickle
import warnings
import arviz as az
import aesara.tensor as at
from pytensor.compile.ops import as_op

warnings.filterwarnings('ignore')

# 定义初始参数
init_pars = [params["PRest"], params["PK"], params["PL"], params["Kbile"], params["GFR"],
             params["Free"], params["Vmax_baso"], params["Km_baso"], params["Kurine"],
             params["Kreab"]]

# 加载优化后的参数
with open('optimized_params.pkl', 'rb') as f:
    popt = pickle.load(f)

# 对初始参数进行对数归一化
initial_guess_normalized = log_normalize(popt)
dose_datas = [pd.DataFrame({'time': [dosing_time_train[idx]], 'Rate': [rate_data_train[idx]]}) for idx in range(len(arm_data_train))]


@as_op(itypes=[at.dvector, at.dvector, at.dmatrix], otypes=[at.dmatrix])
def odeint_op(initial_guess_normalized,time_points_train, dose_datas):
    # 将 TensorVariable 转换为 numpy 数组
    pars =initial_guess_normalized.eval()
    time_points = time_points_train.eval()
    dose_data = dose_datas.eval()
    # 调用 MTX_PBPK 函数
    solution = MTX_PBPK(pars, time_points, dose_data)
    # 返回 numpy 数组
    return np.array(solution)
# 记录开始时间
start_time = time.time()

# 定义 MCMC 优化
# 定义pymc模型
with pm.Model() as model:
    # 定义参数的先验分布，使用initial_guess_normalized的长度作为shape
    pars = pm.Normal('pars', mu=initial_guess_normalized, sigma=1.0, shape=len(initial_guess_normalized))

    # 定义一个空列表来存储每组数据的预测值
    predicted_list = []

    # 循环处理每组数据
    for tp, dd in zip(time_points_train, dose_datas):
        # 将 pandas DataFrame 转换为 numpy 数组
        dd_array = dd.values.astype(np.float64)
        # 计算每组数据的预测值
        predicted = pm.Deterministic(f'predicted_{len(predicted_list)}', odeint_op(pars, tp, dd))
        predicted_list.append(predicted)

    # 将所有预测值拼接在一起
    predicted_all = pm.math.concatenate(predicted_list, axis=0)

    # 定义似然函数
    likelihood = pm.Normal('likelihood', mu=predicted_all, sigma=0.1,
                           observed=np.concatenate(concentration_data_train, axis=0))

    # 使用MCMC进行参数优化
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# 查看结果
az.summary(trace, var_names=['pars'])

# 可视化结果
az.plot_trace(trace, var_names=['pars'])
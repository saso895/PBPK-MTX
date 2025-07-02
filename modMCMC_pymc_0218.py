import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import theano.tensor as tt
import pymc3 as pm
import arviz as az
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from scipy.integrate import solve_ivp

# warnings.filterwarnings('ignore')
# theano.config.cxx = 'g++'

# print('*** Start script ***')
# print(f'{pm.__name__}: v. {pm.__version__}')
# print(f'{theano.__name__}: v. {theano.__version__}')

random_seed=20394

# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, params):
    '''定义微分方程的函数，包含药物点滴输入'''
    # params = tt.reshape(params, (10,))
    # 反归一化参数（从对数空间恢复到原始空间）
    # parms = exp_denormalize(params)

    # PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = params
    PRest=params[0]
    PK=params[1]
    PL=params[2]
    Kbile=params[3]
    GFR=params[4]
    Free=params[5]
    Vmax_baso=params[6]
    Km_baso=params[7]
    Kurine = params[8]
    Kreab=params[9]

    # 确保 input_rate 是标量
    # input_rate = R if t <= T_total else 0
    input_rate = pm.math.switch(t > T_total, 0, R)
    

    # ydot = np.zeros(7)
    ydot0 = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (
                QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate
    ydot1 = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot2 = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (
                Km_baso + y[2] / VK / PK)
    ydot3 = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot4 = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[
        4] * Kurine - Kreab * y[4]
    ydot5 = Kurine * y[4]
    ydot6 = Kbile * y[1]

    return [ydot0,ydot1,ydot2,ydot3,ydot4,ydot5,ydot6]

    # 使用 SciPy 求解 ODE
def solve_ode_with_scipy(params, time_points, T_total, R):
        # 将参数和输入率附加到最后
    params_with_input = np.array(params + [T_total, R])
    sol = solve_ivp(derivshiv, (0, time_points[-1]), y0, t_eval=time_points, args=(params_with_input,))
    return sol.y.T

if __name__ == '__main__':
    T_total = 0
    R = 0
    n_states=7
    n_theta=10

    pars=[]

    # 使用 pymc3 进行 MCMC 采样优化参数
    with pm.Model() as model:
        # 设置参数的先验分布
        PRest = pm.Normal('PRest', mu=init_pars["PRest"], sigma=1)
        PK = pm.Normal('PK', mu=init_pars["PK"], sigma=1)
        PL = pm.Normal('PL', mu=init_pars["PL"], sigma=1)
        Kbile = pm.Normal('Kbile', mu=init_pars["Kbile"], sigma=1)
        GFR = pm.Normal('GFR', mu=init_pars["GFR"], sigma=1)
        Free = pm.Normal('Free', mu=init_pars["Free"], sigma=0.1)
        Vmax_baso = pm.Normal('Vmax_baso', mu=init_pars["Vmax_baso"], sigma=10)
        Km_baso = pm.Normal('Km_baso', mu=init_pars["Km_baso"], sigma=1)
        Kurine = pm.Normal('Kurine', mu=init_pars["Kurine"], sigma=0.01)
        Kreab = pm.Normal('Kreab', mu=init_pars["Kreab"], sigma=0.01)

        # 将参数包装成 Theano 的张量
        params = [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]

        trace=None
        # params = tt.reshape(params, (10,))  # 显式指定形状为 (10,)
        # 遍历每组数据，逐组计算拟合值并添加观测数据的似然函数
        # 定义空列表来存储观测数据
        observed_concentrations = []
        start_time = time.time()
        for i in range(len(time_points_train)):
            pars.append(params)
            time_points = time_points_train[i]
            concentration = concentration_data_train[i]
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            D_total = dose
            T_total = timelen
            R = D_total / T_total

            # 使用 partial 将 R 和 T_total 作为额外参数传递给 derivshiv
            # ode_func_partial = partial(derivshiv, R=R, T_total=T_total)

            # 定义新的 ODE 求解器
            #ode_partial = pm.ode.DifferentialEquation(
            #    func=derivshiv,
            #    times=time_points,
            #    n_states=n_states,
             #   n_theta=n_theta,
             #   t0=0
            #)

            # 使用 SciPy 求解 ODE
            ode_solution = solve_ode_with_scipy(params, time_points, T_total, R)

            # 获取观测数据
            observed_concentrations.append(concentration)

            # 定义似然函数
            pm.Normal(f'concentration_{i}', mu=ode_solution[:, 0], sigma=0.1, observed=concentration)
            # 求解 ODE
            # y = ode_partial(y0=[0]*n_states, theta=params)

            #y = ode_partial(y0=tt.zeros(7), theta=params, extra_args={'R': R, 'T_total': T_total})

            # 计算预测浓度
            # VPlas = tt.constant(VPlas)
            # mu = y[:, 0] / VPlas
            
            # 使用 pm.Deterministic 来保存每一组预测值
            # pm.Deterministic(f'predicted_concentration_{i}', mu)

            # y_obs = pm.Normal(f'y_obs_{i}', mu=mu, sigma=0.1, observed=concentration, )
        end_time = time.time()
        print(f"ODE defining for dataset {i} took {end_time - start_time:.2f} seconds")

        # 使用 NUTS 采样算法进行 MCMC 采样
        trace = pm.sample(5000, chains=2, tune=2000, target_accept=0.9, cores=6, random_seed=random_seed)
        # data = az.from_pymc3(trace=trace)
        #best_params = trace.get_posterior().mean(dim=['chain', 'draw']).to_array().values

        data = az.from_pymc3(trace=trace)
        # 绘制 MCMC 采样后的结果
        
        pm.traceplot(trace)
        plt.show()

        # 打印后验分布的摘要
        print(pm.summary(trace))

        summary = pm.summary(trace)
        
        #P = status.mean(axis=0)

        # 生成后验预测值
        # posterior_predictive = pm.sample_posterior_predictive(trace)

        y_pred = []
        # 从后验分布的摘要中提取均值
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

        #### --- 画图 --- ####
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取子图的行数和列数
        num_groups = len(time_points_train)
        rows = (num_groups + 2) // 3
        cols = 3
        # 创建画布，指定大小
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        # 绘制观测值和后验预测值
        for i in range(len(time_points_train)):
            time_points = time_points_train[i]
            concentration = concentration_data_train[i]
            dose = input_dose_train[i]
            timelen = inject_timelen_train[i]
            D_total = dose
            T_total = timelen
            R = D_total / T_total

            # 使用 partial 将 R 和 T_total 作为额外参数传递给 derivshiv
            # ode_func_partial = partial(derivshiv, R=R, T_total=T_total)

            # 定义新的 ODE 求解器
            ode_partial = pm.ode.DifferentialEquation(
                func=derivshiv,
                times=time_points,
                n_states=n_states,
                n_theta=n_theta,
                t0=0
            )

            # 求解 ODE
            y = ode_partial(y0=[0]*n_states, theta=best_params)
            # 计算预测浓度
            #VPlas = tt.constant(VPlas)
            predicted_concentration = y[:, 0] / VPlas   
            predicted_concentration = predicted_concentration.eval()         
            y_pred.append(y)

            # 在对应的子图上绘制散点和拟合曲线
            axes[i].scatter(time_points, concentration, label=f'训练数据 组 {i+1}', color='red')            
            axes[i].scatter(time_points, predicted_concentration, label=f'拟合数据 组 {i+1}', color='green')            
            axes[i].plot(time_points, predicted_concentration, label=f'拟合曲线 组 {i+1}', color='blue')
            axes[i].set_xlabel('时间 (小时)')
            axes[i].set_ylabel('药物浓度 (ug/mL)')
            axes[i].set_title(f'药物浓度拟合 组 {i+1}')
            axes[i].legend()
            
        # 如果子图数量不足，隐藏多余的子图
        #for j in range(i + 1, len(axes)):
        #    fig.delaxes(axes[j])
        plt.tight_layout()

    # 保存图像为矢量图格式
    plt.savefig('saved_result/fit_results.svg', format='svg')
    plt.show()



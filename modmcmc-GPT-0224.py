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
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
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

# 药代动力学模型函数，用于参数拟合
def pk_model(t, D_total, T_total, *params):
    '''药代动力学模型函数，用于参数拟合'''
    print(f"params : {params}") 
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

if __name__ == '__main__':

    # 使用 pymc3 进行 MCMC 采样优化参数
    with pm.Model() as model:
        # 设置参数的先验分布pm.Normal('PRest', mu=0, sigma=100, dims="sim")
        PRest = pm.Normal('PRest', mu=0, sigma=10)
        PK = pm.Normal('PK', mu=0, sigma=10)
        PL = pm.Normal('PL', mu=0, sigma=10)
        Kbile = pm.Normal('Kbile', mu=0, sigma=10)
        GFR = pm.Normal('GFR', mu=0, sigma=10)
        Free = pm.Normal('Free', mu=0, sigma=10)
        Vmax_baso = pm.Normal('Vmax_baso', mu=0, sigma=10)
        Km_baso = pm.Normal('Km_baso', mu=0, sigma=10)
        Kurine = pm.Normal('Kurine', mu=0, sigma=10)
        Kreab = pm.Normal('Kreab', mu=0, sigma=10)
        # 每一组预测值
        sigma = pm.HalfNormal("sigma", 1)

        params = np.array([init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"], \
                   init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"], init_pars["Kreab"]])
       
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
            mu = pk_model(time_points, D_total, T_total, *params) 
            print(f"mu for y_obs_{i}: {mu}")  # 打印 mu 的值
            y_obs.append(pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration))
            #Y = pm.Normal(f'y_obs_{i}', mu=mu, sigma=sigma, observed=concentration)
        step=pm.NUTS()
        trace = pm.sample(1000, step=step,chains=4, tune=500,  cores=20, discard_tuned_samples=False, random_seed=1)#target_accept=0.9,
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
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # 定义保存路径
        save_dir = f'saved_result/{today_date}'
        os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
        # 保存 y_pred
       
    #     np.save('save_dir', best_params)

        with tqdm(range(len(time_points_train))) as ybar:
            best_params = np.load('saved_result/best_params.npy')
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
                #R = D_total / T_total

                # 求解 ODE
                y = pk_model(time_points, D_total, T_total, *best_params) 
                # 计算预测浓度
                #VPlas = tt.constant(VPlas)
                predicted_concentration = y   
                #predicted_concentration = predicted_concentration.eval()         
                y_pred.append(predicted_concentration)

            # 获取当前日期
            today_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # 定义保存路径
            save_dir = f'saved_result/{today_date}'
            os.makedirs(save_dir, exist_ok=True)  # 如果文件夹已存在，则不会报错
            # 保存 y_pred
            file_path = os.path.join(save_dir, 'y_pred.pkl')
            #file_path = f'saved_dir/y_pred.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(y_pred, f)   

        with open(file_path, 'rb') as f:
            y_pred = pickle.load(f)
        
        #### --- 画图 --- ####
        # with tqdm(range(len(time_points_train))) as pbar:
        #     plt.rcParams['font.sans-serif'] = ['SimHei']
        #     plt.rcParams['axes.unicode_minus'] = False

        #     # 获取子图的行数和列数
        #     num_groups = len(time_points_train)
        #     rows = (num_groups + 2) // 3
        #     cols = 3

        #     # 创建画布，指定大小
        #     fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        #     axes = axes.flatten()
        #     for i in pbar:
        #         pbar.set_description("Predicting sampe: ") # 设置描述
        #         time = time_points_train[i]
        #         concentration = concentration_data_train[i]
        #         dose = input_dose_train[i]
        #         timelen = inject_timelen_train[i]
        #         D_total = dose
        #         T_total = timelen
        #         pred_y=y_pred[i]
                
        #         # 在对应的子图上绘制散点和拟合曲线
        #         axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='red')
        #         axes[i].scatter(time, pred_y, label=f'拟合数据 组 {i+1}', color='green')        
        #         axes[i].plot(time, pred_y, label=f'拟合曲线 组 {i+1}', color='blue')
        #         axes[i].set_xlabel('时间 (小时)')
        #         axes[i].set_ylabel('药物浓度 (ug/mL)')
        #         axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        #         axes[i].legend()
                
        #     # 如果子图数量不足，隐藏多余的子图
        #     for j in range(i + 1, len(axes)):
        #         fig.delaxes(axes[j])

        #     #plt.tight_layout()

        #     # 保存图像为矢量图格式
        #     file_path = os.path.join(save_dir, 'fit_results0220.svg')
        #     plt.savefig(file_path, format='svg')
        #     plt.show()

            all_predicted = np.concatenate(y_pred)
            all_observed = np.concatenate(concentration_data_train)

            # 绘制散点图：x 为预测值，y 为观察值
            plt.scatter(all_predicted, all_observed, color='blue', alpha=0.6, label='Prediction vs Observation')

            # 绘制理想拟合线（y = x），表示完美拟合的情况
            # 绘制理想拟合线（y = x），表示完美拟合的情况
            # min_val = min(min(all_predicted), min(all_observed))
            # max_val = max(max(all_predicted), max(all_observed))
            # plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit')
            # plt.plot([min(all_predicted), max(all_predicted)], 
            #         [min(all_predicted), max(all_predicted)], 
            #         color='red', linestyle='--', label='Ideal Fit')

            # 设置图表标签和标题
            plt.xlabel('Predicted Concentration')
            plt.ylabel('Observed Concentration')
            plt.title('Predicted vs Observed Concentrations (All Patients)')
            plt.legend()
            plt.grid(True)

            # 显示图形
            plt.tight_layout()
            plt.savefig('predicted_vs_observed_all_patients.png')
            plt.show()










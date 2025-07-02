import numpy as np
import matplotlib.pyplot as plt
import emcee
from pk_model1209 import pk_model, derivshiv
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import pickle


# 对参数进行归一化
def normalize_params(params, lower_bounds, upper_bounds):
    """将参数归一化到 0 到 1 的范围内"""
    return (params - lower_bounds) / (upper_bounds - lower_bounds)

def denormalize_params(normalized_params, lower_bounds, upper_bounds):
    """将归一化参数恢复到原始范围"""
    return normalized_params * (upper_bounds - lower_bounds) + lower_bounds

# 修改 log_likelihood 函数，使其使用反归一化的参数
def log_likelihood(normalized_params, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train):
    log_likelihood_val = 0.0

    # 反归一化参数
    # params = denormalize_params(normalized_params, lower_bounds, upper_bounds)
    # PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = params

    for i in range(len(time_points_train)):
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        timelen = inject_timelen_train[i]
        D_total = dose
        T_total = timelen

        # 调用药代动力学模型
        mu = pk_model(time, dose, timelen, *normalized_params)

        # 假设观测数据服从正态分布，标准差为0.1
        log_likelihood_val += -0.5 * np.sum(((concentration - mu) / 0.1) ** 2)

    return log_likelihood_val

# PRest               =  0.2      # Restofbody/plasma PC; (Average of fat and non fat)  
# MW                  =  454.439  # g/mol, MTX molecular mass 
# Free                =  0.58     # MTX unbound fractions in plasma (Giuseppe Pesenti et al., 2021)
# Vmax_baso_invitro   =  242.75   # pmol/mg protein/min, Average of OAT3 and Rfc1
# Km_baso             =  17.814   # mg/L, Km of basolateral transpoter, Average of OAT3 and Rfc1
# protein             =  2.0e-6   # mg protein/proximal tubuel cell, Amount of protein in proximal tubule cells
# GFR                 =  13.9     # L/hr (Kayode Ogungbenro，2014)
# Kreab               =  0.1      # L/hr
# Kbile               =  3.3      # L/hr
# Kurine              =  0.063    # L/h, Rate of urine elimination from urine storage (male) (fit to data)
# 定义先验函数
# log_prior 函数
def log_prior(normalized_params):
    # 反归一化参数
    params = denormalize_params(normalized_params, lower_bounds, upper_bounds)
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = params

    # 定义参数的先验分布范围
    if (0.01 < PRest < 2.0 and 0.01 < PK < 5.0 and 0.01 < PL < 10 and
        0.1 < Kbile < 10.0 and 5 < GFR < 50 and 0.1 < Free < 1.0 and
        120 < Vmax_baso < 200 and 10 < Km_baso < 25 and 0.001 < Kurine < 0.2 and
        0.001 < Kreab < 0.5):
        return 0.0  # 在合理范围内，返回0，表示对数先验概率为0
    return -np.inf  # 否则返回负无穷，表示该参数组合不合理

# 定义后验概率函数
def log_probability(normalized_params, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train):
    lp = log_prior(normalized_params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(normalized_params, time_points_train, concentration_data_train, input_dose_train, inject_timelen_train)
    if not np.isfinite(ll):
        return -np.inf  # 同样确保似然值为有限值
    return lp + ll


# 加载优化的参数
with open('pars/modfit_pars.pkl', 'rb') as f:
    average_params = pickle.load(f)

# 计算优化参数的平均值，作为 MCMC 的初始均值
#average_params = np.mean(optimized_params, axis=0)
#print(f"平均优化参数: {average_params}")

# 参数的上下界
lower_bounds = np.array([0.01, 0.01, 0.01, 0.1, 5, 0.1, 120, 10, 0.001, 0.001])
upper_bounds = np.array([2.0, 5.0, 10.0, 10.0, 50.0, 1.0, 200.0, 25.0, 0.2, 0.5])

# 设置MCMC采样的初始值
nwalkers, ndim = 32, 10  # 32个采样链，10个参数
# 设置 MCMC 采样的初始值，并对其进行归一化
initial_pos = [normalize_params(average_params, lower_bounds, upper_bounds) + 0.1 * np.random.randn(ndim) for i in range(nwalkers)]

# 创建emcee采样器
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(time_points_train, concentration_data_train, input_dose_train, inject_timelen_train))

# 进行MCMC采样
print("Running MCMC...")
sampler.run_mcmc(initial_pos, 20000, progress=True)

# 获取采样结果
samples_normalized = sampler.get_chain(discard=100, thin=15, flat=True)

# 反归一化采样结果
samples = denormalize_params(samples_normalized, lower_bounds, upper_bounds)

# 保存归一化后的采样结果
with open('pars/modmcmc_pars.pkl', 'wb') as f:
    pickle.dump(samples, f)
    
# # 1. 绘制并保存每个参数的 MCMC 采样结果
# fig, axes = plt.subplots(10, figsize=(10, 7), sharex=True)
labels = ["PRest", "PK", "PL", "Kbile", "GFR", "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab"]

# # 遍历每个参数并绘制采样结果
# for i in range(len(labels)):
#     ax = axes[i]
#     ax.plot(samples[:, i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)
# axes[-1].set_xlabel("Step number")
# plt.tight_layout()

# # 保存参数采样结果到文件
# plt.savefig('mcmc_sampling_results.png')
# # plt.show()

# 计算每个参数的均值和标准差
param_means = np.mean(samples, axis=0)
param_stds = np.std(samples, axis=0)

# 打印每个参数的均值和标准差
for i, (mean, std) in enumerate(zip(param_means, param_stds)):
    print(f"{labels[i]}: {mean:.3f} ± {std:.3f}")
    

# 创建图形
plt.figure(figsize=(8, 8))

# 初始化存储所有患者的预测值和观察值的列表
all_predicted = []
all_observed = []

param_means_norm = normalize_params(param_means, lower_bounds, upper_bounds)

# 遍历每个患者的数据
for i in range(len(time_points_train)):
    time = time_points_train[i]
    concentration = concentration_data_train[i]
    dose = input_dose_train[i]
    timelen = inject_timelen_train[i]
    
    # 计算预测值 (使用参数均值)
    predicted_concentration = pk_model(time, dose, timelen, *param_means_norm)  # 使用均值参数进行预测

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
plt.savefig('predicted_vs_observed_all_patients.png')
plt.show()


# # 2. 将所有患者的散点图(子图)绘制到同一张图中
# # N 个患者
# num_patients = len(time_points_train)

# # 设置子图的布局，例如 3 行 4 列（根据患者数量自动调整）
# rows = (num_patients + 3) // 4  # 每行最多4个子图
# cols = min(num_patients, 4)     # 最多4列，防止溢出

# # 创建子图，并确保每个子图的比例接近正方形
# fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

# # 设置不同的颜色和符号来区分不同患者
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
# markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']

# # 遍历患者并在每个子图上绘制对应的散点图
# for i in range(num_patients):
#     time = time_points_train[i]
#     concentration = concentration_data_train[i]
#     dose = input_dose_train[i]
#     timelen = inject_timelen_train[i]
    
#     # 计算预测值 (使用参数均值)
#     predicted_concentration = pk_model(time, dose, timelen, *param_means)  # 使用均值参数进行预测

#     # 在对应的子图上绘制散点图
#     ax = axes[i // cols, i % cols]  # 通过行列索引定位子图
#     ax.scatter(concentration, predicted_concentration, color=colors[i % len(colors)], 
#                marker=markers[i % len(markers)], label=f'Patient {i+1}')
    
#     # 绘制理想拟合线
#     ax.plot([min(concentration), max(concentration)], 
#             [min(concentration), max(concentration)], 
#             color='black', linestyle='--', label='Ideal Fit')

#     ax.set_xlabel('Observed Concentration')
#     ax.set_ylabel('Predicted Concentration')
#     ax.set_title(f'Patient {i+1}')
#     ax.legend()
#     ax.grid(True)

# # 隐藏多余的子图（如果有多余的子图）
# for j in range(i + 1, rows * cols):
#     fig.delaxes(axes[j // cols, j % cols])

# # 调整子图布局，确保不重叠
# plt.subplots_adjust(hspace=0.4, wspace=0.4)

# # 保存到文件
# plt.savefig('all_patients_subplots.png')
# # plt.show()
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
from pk_model import pk_model, cost_function, log_normalize, exp_denormalize
from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
from init_param import init_pars
import warnings

warnings.filterwarnings('ignore')


# 初始猜测的参数，仅包括可优化的参数
initial_guess = [init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"], 
                 init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"], init_pars["Kreab"]]

# 对 initial_guess 进行对数归一化
initial_guess_normalized = log_normalize(initial_guess)

# 对训练集进行拟合
optimized_params = []

for i in range(len(time_points_train)):
    time = time_points_train[i]
    concentration = concentration_data_train[i]
    dose = input_dose_train[i]
    timelen = inject_timelen_train[i]
    D_total = dose
    T_total = timelen
    
    # 使用 minimize 函数进行参数优化
    result = minimize(cost_function, initial_guess_normalized, args=(time, concentration, D_total, T_total), method='L-BFGS-B')
    popt = exp_denormalize(result.x)

    optimized_params.append(popt)
    
    print(f"组 {i+1} 的优化参数: {popt}")
    print("优化结果消息:", result.message)
    print("是否成功:", result.success)
    print("最终目标函数值:", result.fun)
with open('pars/modfit_pars_xw.pkl', 'wb') as f:
    pickle.dump(optimized_params, f)
    

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
for i in range(len(time_points_train)):
    time = time_points_train[i]
    concentration = concentration_data_train[i]
    dose = input_dose_train[i]
    timelen = inject_timelen_train[i]
    D_total = dose
    T_total = timelen
    
    # 在对应的子图上绘制散点和拟合曲线
    axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='red')
    pred_y = pk_model(time, D_total, T_total, *optimized_params[i])
    axes[i].scatter(time, pred_y, label=f'拟合数据 组 {i+1}', color='green')
    
    axes[i].plot(time, pred_y, label=f'拟合曲线 组 {i+1}', color='blue')
    axes[i].set_xlabel('时间 (小时)')
    axes[i].set_ylabel('药物浓度 (ug/mL)')
    axes[i].set_title(f'药物浓度拟合 组 {i+1}')
    axes[i].legend()
    
# 如果子图数量不足，隐藏多余的子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# 保存图像为矢量图格式
plt.savefig('saved_result/fit_results.svg', format='svg')
# plt.show()


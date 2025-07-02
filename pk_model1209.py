import numpy as np
from scipy.integrate import odeint
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import theano.tensor as tt
import pymc3 as pm

from init_data import time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
# 对数归一化和反归一化函数
def log_normalize(params):
    """对参数进行对数归一化"""
    return np.log(params)  # 确保对数变换只对正数操作
    #return np.log(np.array(params))
    #return np.log(np.array(list(params.values())))

def exp_denormalize(log_params):
    """对数归一化的反操作（指数恢复）"""
    return np.exp(log_params)
    #return np.exp(np.array(log_params))
# 定义微分方程的函数，包含药物点滴输入
def derivshiv(y, t, log_params, R, T_total):
    '''定义微分方程的函数，包含药物点滴输入'''
    log_params = tt.reshape(log_params, (10,))
    # 反归一化参数（从对数空间恢复到原始空间）
    #parms = exp_denormalize(log_params)
    
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = log_params
    # 确保 input_rate 是标量
    input_rate = R if t <= T_total else 0

    ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]

    return ydot

# 药代动力学模型函数，用于参数拟合
def pk_model(t, D_total, T_total, *normalized_params):
    '''药代动力学模型函数，用于参数拟合'''
    
    # # 对参数进行对数归一化
    # normalized_params = log_normalize(params)
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)
    
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        t, 
        args=(normalized_params, R, T_total), 
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)
    return y[:, 0] / VPlas

# 自定义的成本函数：计算预测值和观测值之间的平方误差和
call_count = 0
def cost_function(params,  time_points_train, concentration_data_train,input_dose_train,inject_timelen_train):
    '''自定义的成本函数：计算预测值和观测值之间的平方误差和'''
    global call_count
    call_count += 1
    #print(f"Total cost 调用次数: {call_count}")
    total_cost = 0
    for i in range(len(time_points_train)):
        concentration = concentration_data_train[i]
        dose = input_dose_train[i]
        T_total  = inject_timelen_train[i]
        D_total = dose
        timelen=time_points_train[i]
        y_pred = pk_model(timelen, D_total, T_total, *params)
        observed_values = concentration_data_train[i]
        cost = np.sum((concentration - y_pred) ** 2)
        #print(f"组 {idx + 1} 的成本: {cost}")
        total_cost += cost
    return total_cost

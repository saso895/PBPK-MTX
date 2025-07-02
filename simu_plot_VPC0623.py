import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import os
from functools import partial
from init_param import init_pars
import numpy as np
from init_param import QRest, QK, QL, QPlas, VRest, VK, VL, VPlas
import arviz as az
from init_data_point4 import df,time_points_train, concentration_data_train, input_dose_train, inject_timelen_train
import time,datetime
from tqdm import tqdm
import pickle
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

#==========读入模拟数据
with open('207result\chain1_0619_207.pkl', 'rb') as f:
    y_chain1=pickle.load( f)
with open('207result/chain2_0619_207.pkl', 'rb') as f:
    y_chain2=pickle.load( f)
with open('207result/chain3_0619_207.pkl', 'rb') as f:
    y_chain3=pickle.load( f)
with open('207result\chain4_0619_207.pkl', 'rb') as f:
    y_chain4=pickle.load( f)      
with open('207result/chainALL_0619_207.pkl','rb') as f:
    y_chainALL = pickle.load(f)
with open('saved_result/GA_simu0506_0.pkl','rb') as f:
    y_fitga = pickle.load(f)

### --- 画图 --- ####
with tqdm(range(len(time_points_train))) as pbar:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取子图的行数和列数
    num_groups = len(time_points_train)
    rows = (num_groups + 2) // 3
    cols = 3

    # 创建画布，指定大小
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    #all_predicted = []
    # ----------🔵 新增：给 6 条曲线各建一个列表收集所有患者的预测 ----------
    all_chain1      = []
    all_chain2     = []
    all_chain3     = []
    all_chain4  = []   # chain4 → y_fit_GA
    all_chain    = []
    all_FITGA  = []   # GA_simu0506 → y_fitga
#（旧的 all_predicted 不用了，可删除）

    for i in pbar:
        pbar.set_description("Predicting sampe: ") # 设置描述
        time = time_points_train[i]
        concentration = concentration_data_train[i]
        chain1=y_chain1[i]
        chain2=y_chain2[i]
        chain3=y_chain3[i]
        chain4=y_chain4[i]
        chain=y_chainALL[i]
        #GOF_FIT=y_GOF[i]
        FITGA_y=y_fitga[i]
        #======搜集GOF数据=========#
        # ------🟠 改动：对 6 条曲线各插值一次 ------
        # pred_1     = np.interp(time, chain1[:,0], chain1[:,1])
        # pred_2     = np.interp(time, chain2[:,0], chain2[:,1])
        # pred_3     = np.interp(time, chain3[:,0], chain3[:,1])
        # pred_4     = np.interp(time, chain4[:,0], chain4[:,1])   # chain4
        # pred_all   = np.interp(time, chain[:,0],  chain[:,1])
        # pred_FITGA = np.interp(time, FITGA_y[:,0], FITGA_y[:,1])     # fi
        # --- ① 汇总 6 条预测到 2D 数组：(n_curve, n_time)
        #preds = np.vstack([pred_1, pred_2, pred_3, pred_4, pred_all, pred_FITGA])

        # --- ② 计算 5th / 95th 百分位
        p5  = chain1*0.8
        p95 = chain1*1.2

        # # 追加到各自“全局”列表
        # all_chain1.extend(pred_1)
        # all_chain2.extend(pred_2)
        # all_chain3.extend(pred_3)
        # all_chain4.extend(pred_4)
        # all_chain.extend(pred_all)
        # all_FITGA.extend(pred_FITGA)

        # 在对应的子图上绘制散点和拟合曲线
        axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='#E73235')    
        axes[i].plot(chain1[:,0], chain1[:,1], label=f'chain1曲线 组 {i+1}', color='#fdd363',lw=1)
        axes[i].plot(chain1[:,0], chain1[:,1]*0.8, '--', label='5%分位数', color='blue', alpha=0.6)
        axes[i].plot(chain1[:,0], chain1[:,1]*1.2, '--', label='95%分位数', color='blue', alpha=0.6)          
        # axes[i].plot(chain2[:,0], chain2[:,1], label=f'chain2拟合曲线 组 {i+1}', color='#5ca788')        
        # axes[i].plot(chain3[:,0], chain3[:,1], label=f'chain3曲线 组 {i+1}', color='#227abc')
        # axes[i].plot(chain4[:,0], chain4[:,1], label=f'chain4曲线 组 {i+1}', color='#b96d93')
        # axes[i].plot(chain[:,0], chain[:,1], label=f'all曲线 组 {i+1}', color='#E73235')
        # axes[i].plot(FITGA_y[:,0], FITGA_y[:,1], label=f'fitGA曲线 组 {i+1}', color='#9467bd')
        axes[i].set_xlabel('时间 (小时)')
        axes[i].set_ylabel('药物浓度 (mg/L)')
        axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        axes[i].legend()

        
    # 如果子图数量不足，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()

# 保存图像为矢量图格式
save_path =f'207result/Simuplot_{today_date}_95.svg'
plt.savefig(save_path, format='svg')
plt.show()

    

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
from sklearn.preprocessing import MinMaxScaler  # ✅ 新增，解决 Pylance 报错
from matplotlib.lines import Line2D 

today_date = datetime.datetime.now().strftime('%Y-%m-%d')
BASE_DIR   = 'saved_result'
DATA_NAME  = 'simu01_modfit'
#==========读入模拟数据
with open(f'{BASE_DIR}\{DATA_NAME}_{today_date}.pkl', 'rb') as f:
    y_simu=pickle.load( f)

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
    # === 初始化误差收集容器 ===
    result_rows = []  # [Patient_ID, AFE, AAFE, CP90, Tag]
    good_ids = []
    for i in pbar:
        pbar.set_description("Predicting sampe: ") # 设置描述
        time = time_points_train[i]
        concentration = concentration_data_train[i]

        # === ✂️ 过滤 t = 0 h 的观测点 ===
        mask_t = time > 0              # True / False 数组
        time          = time[mask_t]   # 保留 t > 0 的采样时刻
        concentration = concentration[mask_t]
        if len(time) < 2:              # 极端：只剩 1 个点 → 跳过该病人
            print(f"⚠️ 病人 {i+1} 仅剩 0 个有效点，已跳过")
            continue
        y=y_simu[i]
        # 在对应的子图上绘制散点和拟合曲线
        axes[i].scatter(time, concentration, label=f'训练数据 组 {i+1}', color='#E73235')    
        axes[i].plot(y[:,0], y[:,1], label=f'预测曲线 组 {i+1}', color='#fdd363',lw=1)
        axes[i].plot(y[:,0], y[:,1]*0.8, '--', label='5%分位数', color='blue', alpha=0.6)
        axes[i].plot(y[:,0], y[:,1]*1.2, '--', label='95%分位数', color='blue', alpha=0.6)          
        axes[i].set_xlabel('时间 (小时)')
        axes[i].set_ylabel('药物浓度 (mg/L)')
        axes[i].set_title(f'药物浓度拟合 组 {i+1}')
        axes[i].legend()
        
        # === 🟡 误差指标分析（y 为基准） =======================
        y_obs = concentration
        # 使用插值将 y 预测值映射到观测时间点
        y_pred = np.interp(time, y[:, 0], y[:, 1])
        y_5 = y_pred * 0.8
        y_95 = y_pred * 1.2
        fold_err = y_pred / y_obs
        log_fe = np.log10(fold_err)
        # --- AFE & AAFE
        afe = 10 ** np.mean(log_fe)
        aafe = 10 ** np.mean(np.abs(log_fe))
        # --- CP90
        cp90 = np.mean((y_obs >= y_5) & (y_obs <= y_95))

        # --- 标签规则（依据文献：AFE 与 AAFE 均在 0.5–2 fold 内视为 good）
        if 0.5 <= aafe <= 2 and 0.5 <= afe <= 2:
            tag = "good"
            good_ids.append(i + 1) 
        else:
            tag = "poor"
                # 保存到结果
        result_rows.append([i + 1, afe, aafe, cp90, tag])

        # ---- 增强图例 ----
        extra_label = f"AFE={afe:.2f}, AAFE={aafe:.2f}, Tag={tag}"
        handles, labels = axes[i].get_legend_handles_labels()
        dummy_handle = Line2D([], [], color="none", label=extra_label)
        handles.append(dummy_handle)
        labels.append(extra_label)
        axes[i].legend(handles, labels, loc="upper right")

    # 如果子图数量不足，隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()

# === 保存拟合图 ==================================================
save_path =f'{BASE_DIR}/Simuplot_{DATA_NAME}_{today_date}.svg'
plt.savefig(save_path, format='svg')
plt.show()

# === 🔵 结果输出为 Excel 表格 ====================================
df_result = pd.DataFrame(result_rows, columns=["Patient_ID", "AFE", "AAFE", "CP90", "Tag"])
excel_path = f'{BASE_DIR}/Patient_Errors_{DATA_NAME}_{today_date}.xlsx'
df_result.to_excel(excel_path, index=False)
print(f"✅ 每病人误差评分结果已保存: {excel_path}")

# === 绘制热图（AFE / AAFE / CP90） ============================================
# 若有 NaN 先填列最大值，防止归一化报错
for col in ['AFE', 'AAFE', 'CP90']:
    if df_result[col].isna().all():
        df_result[col] = 0
    else:
        df_result[col].fillna(df_result[col].max(), inplace=True)

hm_data = pd.DataFrame(
    MinMaxScaler().fit_transform(df_result[['AFE', 'AAFE', 'CP90']]),
    columns=['AFE', 'AAFE', 'CP90'],
    index=df_result['Patient_ID']
)

plt.figure(figsize=(8, 6))
plt.imshow(hm_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized 0–1')
plt.xticks(range(3), ['AFE', 'AAFE', 'CP90'])
plt.yticks(range(len(hm_data.index)), hm_data.index)
plt.title('Per‑Patient Prediction Metrics (0–1 normalized)')
plt.xlabel('Metric')
plt.ylabel('Patient ID')
plt.tight_layout()
heatmap_path = f'{BASE_DIR}/Heatmap_{DATA_NAME}_{today_date}.png'
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"✅ 热图已保存: {heatmap_path}")

# ③   保存 good 病人 ID 清单                # === NEW ===
good_id_path = f'{BASE_DIR}/good_patient_{DATA_NAME}_{today_date}.txt'
with open(good_id_path, 'w', encoding='utf-8') as f:
    f.write(','.join(map(str, good_ids)))
print(f"✅ good 病人 ID 已保存: {good_id_path}")
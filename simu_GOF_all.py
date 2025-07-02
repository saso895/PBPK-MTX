#!/usr/bin/env python
# gof_good_patients.py  <– 文件名可自取
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import pickle, datetime, os, glob
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from init_data_point4 import (time_points_train,
                              concentration_data_train)

# ----------- 参数区：路径和文件名可按需调整 -----------
today       = datetime.datetime.now().strftime('%Y-%m-%d')
server =209
result_dir  = f'{server}result'
fixed_name  = f'good_patient_ids_{today}.txt'

# ------------ ① 找到 ID 文件（自动兜底） ---------------  # === FIX ===
if os.path.isfile(os.path.join(result_dir, fixed_name)):
    good_id_file = os.path.join(result_dir, fixed_name)
else:
    # 如果今天的文件还没有，就找最新的一份
    candidates = sorted(glob.glob(os.path.join(result_dir,
                                               'good_patient_ids_*.txt')))
    if not candidates:
        raise FileNotFoundError("❌ 没找到 good_patient_ids_*.txt，请先跑生成脚本")
    good_id_file = candidates[-1]          # 最新的一份

print(f"📑 读取 good 病人 ID 文件: {os.path.basename(good_id_file)}")

# ------------ ② 读取并过滤空字符串 ----------------------  # === FIX ===
with open(good_id_file, 'r', encoding='utf-8') as f:
    txt = f.read().replace('\n', ',')      # 支持换行或逗号
good_ids = [int(x) for x in txt.split(',') if x.strip()]

if not good_ids:
    raise RuntimeError("⚠️ good 病人列表为空，确认生成脚本输出是否正常")

# --------- 载入 chain1 预测结果（按需替换文件名） --------
with open(os.path.join(result_dir, 'chain1_0620_209.pkl'), 'rb') as f:
    y_chain1 = pickle.load(f)

# ---------- 聚合所有 good 病人的 观测 / 预测 -----------
all_obs, all_pred = [], []

for pid in good_ids:
    idx   = pid - 1                         # 病人索引（从 0 起）
    t_obs = time_points_train[idx]
    c_obs = concentration_data_train[idx]

    # 过滤 t=0
    mask  = t_obs > 0
    t_use = t_obs[mask]
    c_use = c_obs[mask]

    if len(t_use) < 2:                      # 观测点太少直接跳过
        continue

    chain1 = y_chain1[idx]                 # shape (N, 2): [time, conc]
    c_hat  = np.interp(t_use, chain1[:, 0], chain1[:, 1])

    all_obs.append(c_use)
    all_pred.append(c_hat)

# 平铺为 1-D 数组
all_obs  = np.concatenate(all_obs)
all_pred = np.concatenate(all_pred)
min_positive = 1e-3
glob_min = min(all_pred.min(), all_obs.min())
glob_min = max(min_positive, glob_min)
glob_max = max(all_pred.max(), all_obs.max())
pad_low  = 0.5   # 对应 10^(-0.5) ≈ ×0.32
pad_high = 0.5   # 对应 10^(+0.5) ≈ ×3.16
x_min = glob_min * (10**-pad_low)
x_max = glob_max * (10** pad_high)

# ---------- 计算 R² 并绘制 GOF -------------------------
r2 = r2_score(all_obs, all_pred)

plt.figure(figsize=(6, 6))
plt.scatter(all_obs, all_pred, alpha=0.6)
max_val = max(all_obs.max(), all_pred.max())
plt.plot([x_min, x_max], [x_min, x_max], 'r--', lw=1,color='red')   # 全局对角线
plt.plot([x_min, x_max], [x_min*0.5, x_max*0.5], 'r--', lw=1, alpha=0.6,color='blue')  # 0.5×
plt.plot([x_min, x_max], [x_min*2.0,  x_max*2.0 ], 'r--', lw=1, alpha=0.6,color='blue')  # 2×
# === 2-fold FIX ===  把 x、y 轴都设为对数尺度
plt.xscale('log')
plt.yscale('log')
# plt.plot([0, max_val], [0, max_val], ls='--', lw=1,color='red')

# # === 2-fold === 上界：y = 2x
# plt.plot([0, max_val], [0, max_val/2], ls='--', lw=1,color='blue')
# # === 2-fold === 下界：y = 0.5x
# plt.plot([0, max_val], [0, max_val*2], ls='--', lw=1,color='blue')
plt.xlabel('Observed Concentration (mg/L)')
plt.ylabel('Predicted Concentration (mg/L)')
plt.title(f'GOF Plot for GOOD Patients ({len(good_ids)} subjects)')

plt.text(0.05, 0.95,
         f'$R^2 = {r2:.3f}$',
         transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', alpha=0.15))

plt.legend()
plt.tight_layout()

out_png = os.path.join(result_dir,
                       f'GOF_good_{today}.png')
plt.savefig(out_png, dpi=300)
plt.show()
print(f"✅ GOF 图已保存: {out_png}")

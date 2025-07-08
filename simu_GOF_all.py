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
result_dir  = f'saved_result'   #{server}result
DATA_NAME   = 'simu01_modfit'   #simu_file
fixed_name  = f'good_patient_{DATA_NAME}_{today}.txt'
SELECT_MODE = "all"        # <<< "all" | "good"  二选一  ### >>> NEW
# ------------ ① 找到 ID 文件（自动兜底） ---------------  # === FIX ===
simu_pkl     = os.path.join(result_dir, f"{DATA_NAME}_{today}.pkl")   ### >>> NEW
good_txt_pat = os.path.join(result_dir, f"good_patient_{DATA_NAME}_{today}.txt")

# ========== ① 读取 SIMU 预测结果 =========================
if not os.path.isfile(simu_pkl):
    raise FileNotFoundError(f"❌ 找不到 {simu_pkl}，请先运行 Simu.py 生成预测文件")
with open(simu_pkl, "rb") as f:               ### >>> NEW
    y_simu = pickle.load(f)                   ### >>> NEW

# ========== ② 获取病人 ID 列表 ==========================
if SELECT_MODE.lower() == "good":             ### >>> NEW
    # 若当天的 good 文本不存在 → 找最近一份
    if not os.path.isfile(good_txt_pat):
        cand = sorted(glob.glob(
            os.path.join(result_dir, f"good_patient_{DATA_NAME}_*.txt")))
        if not cand:
            raise FileNotFoundError("❌ 未找到任何 good_patient_*.txt，"
                                    "请先跑 simu_plot_VPC_every.py")
        good_txt_pat = cand[-1]
    print(f"📑 读取 good 病人 ID 文件: {os.path.basename(good_txt_pat)}")
    with open(good_txt_pat, "r", encoding="utf-8") as f:
        txt = f.read().replace("\n", ",")
    id_list = [int(x) for x in txt.split(",") if x.strip()]
else:                                         ### >>> NEW
    id_list = list(range(1, len(y_simu) + 1)) ### >>> NEW
    print(f"📑 选择所有病人，共 {len(id_list)} 名")        ### >>> NEW

# ---------- 聚合所有 good 病人的 观测 / 预测 -----------
all_obs, all_pred = [], []

for pid in id_list:
    idx   = pid - 1                         # 病人索引（从 0 起）
    t_obs = time_points_train[idx]
    c_obs = concentration_data_train[idx]

    # 过滤 t=0
    mask  = t_obs > 0
    t_use = t_obs[mask]
    c_use = c_obs[mask]

    if len(t_use) < 2:                      # 观测点太少直接跳过
        continue

    y_hat = y_simu[idx]
    c_hat  = np.interp(t_use, y_hat[:, 0], y_hat[:, 1])

    all_obs.append(c_use)
    all_pred.append(c_hat)

# 平铺为 1-D 数组
all_obs  = np.concatenate(all_obs)
all_pred = np.concatenate(all_pred)
min_positive = 1e-3
glob_min = max(all_pred.min(), all_obs.min(),min_positive)
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
plt.xlabel('Observed Concentration (mg/L)')
plt.ylabel('Predicted Concentration (mg/L)')
plt.title(f'GOF Plot for GOOD Patients ({len(id_list)} subjects)')

plt.text(0.05, 0.95,
         f'$R^2 = {r2:.3f}$',
         transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', alpha=0.15))

plt.legend()
plt.tight_layout()

out_png = os.path.join(result_dir,
                       f'GOF_{DATA_NAME}_{SELECT_MODE}{today}.png')
plt.savefig(out_png, dpi=300)
plt.show()
print(f"✅ GOF 图已保存: {out_png}")

# ==============================================================
#  Simu.py  ✧  MCMC‑chain prediction version  (FIXED 2025‑07‑24)
#  --------------------------------------------------------------
#  本文件整合并修复了先前尺寸不匹配 (ValueError) 的问题：
#  • 统一使用 Time_sim = np.arange(0, Duration+0.1, 0.1) 作为模拟
#    时间轴，确保与 mean_pred / p5 / p95 维度一致。
#  • 其他逻辑保持不变，仅在关键位置标记 “🔧 FIX”.
# ==============================================================

# ============================== Simu.py ===============================
import matplotlib.pyplot as plt
import datetime, os, pickle, numpy as np
from tqdm import tqdm
from init_data_point4 import (
    df, time_points_train, concentration_data_train,
    input_dose_train, inject_timelen_train,
)
from init_param import init_pars
from ode_core import PK_model

# === 参数来源设置  ⚠️ 请根据实际文件名修改 ===========================
PARAM_SOURCE = "mcmc"                            # {"init", "modfit", "mcmc", "file"}
CHAIN_IDX = 4 
CHAIN_DRAW_FILE = f"saved_result/chain{CHAIN_IDX}_draws2025-07-19.pkl"  # ← 指向某一条链的抽样
#CHAIN_IDX = 3                                    # 与 DRAW_FILE 对应
N_SAMPLES = 500                                  # 使用多少条样本预测

# === 加载 MCMC 抽样 ----------------------------------------------------
with open(CHAIN_DRAW_FILE, "rb") as f:
    chain_draws = pickle.load(f)                 # ndarray (n_draw, n_param+?)

# 只保留前 10 个模型参数（忽略 sigma 等）
param_draws = chain_draws[:, :10]
# --- 简单抽样（burn‑in=前 10%） --------------------------------------
start = int(0.1 * len(param_draws))
param_draws = param_draws[start:]
if len(param_draws) > N_SAMPLES:
    idx = np.linspace(0, len(param_draws) - 1, N_SAMPLES, dtype=int)
    param_draws = param_draws[idx]

# === 逐病人预测并统计分位数 -----------------------------------------
patient_preds = []   # 每元素: ndarray (len(t), 4) [t, mean, p5, p95]

for i in tqdm(range(len(time_points_train)), desc="Simulating patients"):
    t_obs   = time_points_train[i]
    D_total = input_dose_train[i]
    T_total = inject_timelen_train[i]
    Duration = t_obs[-1]

    # 🔧 FIX: 使用统一细网格 Time_sim，确保与预测数组同长度 ---------
    Time_sim = np.arange(0, Duration + 0.1, 0.1)

    # --- 批量预测 (samples × time) -----------------------------------
    preds = np.vstack([
        PK_model(Time_sim, D_total, T_total, Duration, *p)[:, 1]
        for p in param_draws
    ])  # shape (N_SAMPLES, len(Time_sim))

    mean_pred = preds.mean(axis=0)
    p5_pred   = np.percentile(preds, 5, axis=0)
    p95_pred  = np.percentile(preds, 95, axis=0)

    patient_preds.append(
        np.column_stack([Time_sim, mean_pred, p5_pred, p95_pred])
    )

# === 保存 --------------------------------------------------------------
today_date = datetime.datetime.now().strftime("%Y-%m-%d")
SAVE_DIR   = "saved_result"
os.makedirs(SAVE_DIR, exist_ok=True)
file_tag   = f"simu_mcmc_chain{CHAIN_IDX}_{today_date}.pkl"
with open(os.path.join(SAVE_DIR, file_tag), "wb") as f:
    pickle.dump(patient_preds, f)
print(f"✔ 预测结果已保存 ➜ {file_tag}")

# ==================== simu_plot_VPC_every.py ===========================
import matplotlib.pyplot as plt
import datetime, pickle, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
from init_data_point4 import (
    time_points_train, concentration_data_train,
)

# === 若有额外依赖 (r2_score / mean_squared_error) 可按需导入 ==========


# === 读取预测结果 -----------------------------------------------------
today_date = datetime.datetime.now().strftime("%Y-%m-%d")
BASE_DIR   = "saved_result"
DATA_NAME  = f"simu_mcmc_chain{CHAIN_IDX}"       # 与上面保持一致

with open(f"{BASE_DIR}/{DATA_NAME}_{today_date}.pkl", "rb") as f:
    y_simu = pickle.load(f)

# === 绘图 -------------------------------------------------------------
plt.rcParams.update({"font.sans-serif": ["SimHei"], "axes.unicode_minus": False})
num_groups = len(time_points_train)
rows, cols = (num_groups + 2) // 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

result_rows, good_ids = [], []
for i in tqdm(range(num_groups), desc="Plotting"):
    t_obs   = time_points_train[i]
    c_obs   = concentration_data_train[i]

    mask_t  = t_obs > 0
    t_obs   = t_obs[mask_t]
    c_obs   = c_obs[mask_t]

    if len(t_obs) < 2:
        continue

    y_pred_all = y_simu[i]               # ndarray (t,4)
    t_pred     = y_pred_all[:, 0]
    mean_pred  = y_pred_all[:, 1]
    p5_pred    = y_pred_all[:, 2]
    p95_pred   = y_pred_all[:, 3]

    ax = axes[i]
    ax.scatter(t_obs, c_obs, label=f"观测 组 {i+1}", color="#E73235")
    ax.plot(t_pred, mean_pred, label="平均预测", color="#3762f5")
    ax.plot(t_pred, p5_pred,  "--", label="5th %", color="gray")
    ax.plot(t_pred, p95_pred, "--", label="95th %", color="gray")
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("浓度 (mg/L)")
    ax.set_title(f"浓度预测 组 {i+1}")

    # === 误差指标 =====================================================
    y_pred_mean = np.interp(t_obs, t_pred, mean_pred)
    y_p5        = np.interp(t_obs, t_pred, p5_pred)
    y_p95       = np.interp(t_obs, t_pred, p95_pred)

    fold_err = y_pred_mean / c_obs
    log_fe   = np.log10(fold_err + 1e-12)      # 避免 log(0)
    afe  = 10 ** np.mean(log_fe)
    aafe = 10 ** np.mean(np.abs(log_fe))
    cp90 = np.mean((c_obs >= y_p5) & (c_obs <= y_p95))

    tag = "good" if (aafe <= 2 and 0.5 <= afe <= 2) else "poor"
    if tag == "good":
        good_ids.append(i + 1)

    result_rows.append([i + 1, afe, aafe, cp90, tag])

    extra_label = f"AFE={afe:.2f}, AAFE={aafe:.2f}, Tag={tag}"
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="none", label=extra_label))
    labels.append(extra_label)
    ax.legend(handles, labels)

# 隐藏多余子图 ----------------------------------------------------------
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("PBPK Prediction – Mean & 5/95 Percentiles", y=0.995)
plot_path = f"{BASE_DIR}/Simuplot_{DATA_NAME}_{today_date}.svg"
plt.savefig(plot_path)
plt.show()
print(f"✅ 情景图已保存: {plot_path}")

# === 可选：保存 Excel & Heatmap（与原脚本逻辑一致，可按需启用） =======
# ---------------------------------------------------------------------
# ③   保存 good 病人 ID 清单                # === NEW ===
good_id_path = f'{BASE_DIR}/good_patient_{DATA_NAME}_{today_date}.txt'
with open(good_id_path, 'w', encoding='utf-8') as f:
    f.write(','.join(map(str, good_ids)))
print(f"✅ good 病人 ID 已保存: {good_id_path}")
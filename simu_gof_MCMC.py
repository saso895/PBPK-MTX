# ==============================================================
#  simu_gof.py  —  全局 GOF 图（单独文件）
#  --------------------------------------------------------------
#  使用 Simu.py 生成的预测 pkl，聚合均值预测对全部（或优选）病人
#  观测数据，绘制对数对角线 + 0.5×/2× 边界，输出 PNG。
# ==============================================================
import os, glob, datetime, pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from init_data_point4 import time_points_train, concentration_data_train

# === 配置 =============================================================
CHAIN_IDX    = 4                    # 与 Simu.py 保持一致
SELECT_MODE  = "all"               # "all" 或 "good"
SAVE_DIR     = "saved_result"
TODAY        = datetime.date.today()
PRED_PKL     = f"simu_mcmc_chain{CHAIN_IDX}_{TODAY}.pkl"  # 默认用当天生成的 pkl

# === 读取预测 =========================================================
with open(os.path.join(SAVE_DIR, PRED_PKL), "rb") as f:
    y_simu = pickle.load(f)

# === 可选：只用 good 病人列表 ==========================================
if SELECT_MODE == "good":
    pattern = os.path.join(SAVE_DIR, f"good_patient_simu_mcmc_chain{CHAIN_IDX}_*.txt")
    cand = sorted(glob.glob(pattern))
    if not cand:
        raise FileNotFoundError("❌ 未找到 good_patient_*.txt，请先在 simu_plot.py 中生成")
    good_txt = cand[-1]
    id_list = [int(x) for x in open(good_txt).read().replace("\n", ",").split(",") if x.strip()]
    print(f"📑 从 {os.path.basename(good_txt)} 读取 {len(id_list)} 名 good 病人")
else:
    id_list = list(range(1, len(y_simu)+1))
    print(f"📑 选择所有病人，共 {len(id_list)} 名")

# === 聚合观测 / 预测 ==================================================
all_obs, all_pred = [], []
for pid in id_list:
    idx = pid - 1
    t_obs = time_points_train[idx]
    c_obs = concentration_data_train[idx]
    mask  = t_obs > 0
    t_use, c_use = t_obs[mask], c_obs[mask]
    if len(t_use) < 2:
        continue
    y_hat = y_simu[idx]
    c_hat = np.interp(t_use, y_hat[:,0], y_hat[:,1])   # 取 mean 列
    all_obs.append(c_use)
    all_pred.append(c_hat)
all_obs  = np.concatenate(all_obs)
all_pred = np.concatenate(all_pred)

# === 绘图 =============================================================
r2 = r2_score(all_obs, all_pred)
min_positive = 1e-3
x_min = max(all_obs.min(), all_pred.min(), min_positive) * (10**-0.5)
x_max = max(all_obs.max(), all_pred.max()) * (10**0.5)

plt.figure(figsize=(6,6))
plt.scatter(all_obs, all_pred, alpha=0.6)
plt.plot([x_min, x_max], [x_min, x_max], 'k--', lw=1)
plt.plot([x_min, x_max], [x_min*0.5, x_max*0.5], 'b--', alpha=0.6)
plt.plot([x_min, x_max], [x_min*2.0, x_max*2.0], 'b--', alpha=0.6)
plt.xscale('log'); plt.yscale('log')
plt.xlim(x_min, x_max); plt.ylim(x_min, x_max)
plt.xlabel('Observed Concentration (mg/L)')
plt.ylabel('Predicted Concentration (mg/L)')
plt.title(f'GOF – Chain {CHAIN_IDX} ({SELECT_MODE})  R²={r2:.3f}')
plt.tight_layout()
GOF_PATH = os.path.join(SAVE_DIR, f'GOF_chain{CHAIN_IDX}_{SELECT_MODE}_{TODAY}.png')
plt.savefig(GOF_PATH, dpi=300)
plt.show()
print(f"✅ GOF 图已保存: {GOF_PATH}")

# ---------- Simu_chain1_draws.py  (robust v4) ----------
import pickle, numpy as np, os, textwrap
from tqdm import tqdm
from init_data_point4 import (time_points_train,
                              input_dose_train,
                              inject_timelen_train)
from Simu0623 import pk_model          # ← 你的 PBPK 模型函数

PARAM_PKL = os.path.join('207result', 'chain1_params.pkl')
OUT_PKL   = os.path.join('207result', 'chain1_draw100.pkl')
N_DRAW    = 100

with open(PARAM_PKL, 'rb') as f:
    theta_raw = pickle.load(f)

# ---------- 万能格式探测器 --------------------------------
def _first_finite_2d(obj):
    """
    递归挖出 shape=(N, ≥5) 的二维 float ndarray.
    支持:
        ndarray 2-D
        ndarray 1-D object -> treat as list
        list of ndarray / list / dict
        dict with key 'samples' / 'params' / etc.
    """
    # --- ndarray 直接数值矩阵 ------------
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[0] >= 10 and obj.shape[1] >= 5:
            return obj.astype(float)
        # 1-D object ndarray -> 递归展开
        if obj.ndim == 1 and obj.dtype == object:
            return _first_finite_2d(list(obj))

    # --- list 类型 -----------------------
    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        # list of ndarray
        if isinstance(obj[0], np.ndarray):
            mat = np.vstack(obj)
            if mat.ndim == 2:
                return mat.astype(float)
        # list of list/tuple
        if np.issubdtype(type(obj[0]), (list, tuple)):
            return np.asarray(obj, dtype=float)
        # list of dict
        if isinstance(obj[0], dict):
            param_order = ['PRest', 'PK', 'PL', 'Kbile', 'GFR',
                           'Free', 'Vmax_baso', 'Km_baso', 'Kurine', 'Kreab']
            try:
                mat = np.asarray([[d[k] for k in param_order] for d in obj], dtype=float)
                return mat
            except KeyError:
                pass

    # --- dict 类型 -----------------------
    if isinstance(obj, dict):
        # 优先常见 key
        for k in ['samples', 'params', 'draws', 'theta']:
            if k in obj:
                cand = _first_finite_2d(obj[k])
                if cand is not None:
                    return cand
        # 遍历所有 value
        for v in obj.values():
            cand = _first_finite_2d(v)
            if cand is not None:
                return cand
    return None

theta_pool = _first_finite_2d(theta_raw)

if theta_pool is None:
    print("❌ 仍无法识别结构，请把下列诊断信息给我：")
    import pprint, sys
    pprint.pprint(theta_raw if isinstance(theta_raw, dict) else theta_raw[:3], depth=2)
    sys.exit(1)

print(f"✔ 参数矩阵 shape = {theta_pool.shape}")

# ---------- 随机抽 100 组参数 ------------------------------
if theta_pool.shape[0] < N_DRAW:
    raise RuntimeError(f"样本总数 {theta_pool.shape[0]} < N_DRAW={N_DRAW}")
sel = np.random.choice(theta_pool.shape[0], N_DRAW, replace=False)
theta_bank = theta_pool[sel]

# ---------- 批量模拟 --------------------------------------
all_pred, all_time = [], []
print("▶ 开始 100×每病人模拟...")
for pid, t_obs in enumerate(tqdm(time_points_train, desc="Patients")):
    dose   = input_dose_train[pid]
    tinf   = inject_timelen_train[pid]
    dur    = t_obs[-1]
    tgrid  = np.arange(0, dur+0.1, 0.1)

    curves = [pk_model(tgrid, dose, tinf, dur, *th)[:,1] for th in theta_bank]
    all_pred.append(np.vstack(curves))   # (100, n_time)
    all_time.append(tgrid)

# ---------- 保存 ------------------------------------------
os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
with open(OUT_PKL, 'wb') as f:
    pickle.dump({'time': all_time, 'pred': all_pred}, f)

print(f"✔ 已生成 {OUT_PKL}，可用于 5–95 % dotted line 绘制")
# ----------------------------------------------------------

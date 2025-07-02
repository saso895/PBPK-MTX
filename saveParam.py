import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# 1️⃣  把后验均值抄进列表，顺序必须是
#     [PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]
theta_post_mean = [
    0.01188,     # PRest
    15.15,       # PK
    1.762,       # PL
    0.6791,      # Kbile
    1.81e-10,    # GFR
    159.6,       # Free
    203.4,       # Vmax_baso
    118.3,       # Km_baso
    0.03294,     # Kurine
    0.002271     # Kreab
]

# 2️⃣  转成 numpy 数组（Simu.py 里就是这么用的）
baseline = np.asarray(theta_post_mean, dtype=float)

# 3️⃣  保存 —— 路径与 Simu.py 保持一致即可
save_root = Path("saved_result")
save_root.mkdir(exist_ok=True)

fname = save_root / f"optimized_params{datetime.now():%m%d}_MCMC.pkl"
with open(fname, "wb") as f:
    pickle.dump({"baseline": baseline}, f)   # 用 dict 包一层，Simu.py 能直接识别

print(f"✅ 已保存到 {fname}")

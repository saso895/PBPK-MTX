"""
pbpk_sens_modfit.py —— 基于 ‘胡波湖泊模型’ 三步法的
                       MTX–PBPK 参数敏感性 & 共线性分析
-------------------------------------------------------------------
依赖: numpy ≥1.20, scipy ≥1.6, matplotlib ≥3.3
放置: 与 init_param.py 及 init_data_point4.py 同目录
运行: python pbpk_sens_modfit.py
输出:
  ▸ sens_ranking.png  —— 各参数全局灵敏度条形图
  ▸ gamma_ranges.png —— “子集大小 vs Y 范围” (文献 Fig 1)
  ▸ 终端            —— 灵敏度排序 + 规模 2/3 最坏子集
"""
import itertools, math, os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --------------------------------------------------------------
# 0. 载入 “固定/可调” 参数 & 数据
# --------------------------------------------------------------
from init_param import (  # 固定的生理流量/容积等
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
    PRest, PK, PL, Kbile, GFR, Free,
    Vmax_baso, Km_baso, Kurine, Kreab
)
#   ↑ 可调的 10 个参数与 modfit 顺序保持一致

# 读入 77 例病人（或 split 后训练集）给药方案与采样时间
from init_data_point4 import (          # ← 文件里早已做好预处理:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    time_points_train   as time_groups,     # List[np.ndarray]  每组采样时刻 (h)
    input_dose_train    as dose_groups,     # List[float]       总剂量 (mg)
    inject_timelen_train as tinf_groups     # List[float]       注射时长 (h)
)
# 如需用全部 77 例而非训练集，可把上面三行改为 time_points / input_dose / inject_timelen

# 待分析 10 个参数的名称与基线值
param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free",  "Vmax_baso", "Km_baso", "Kurine", "Kreab"
]
baseline = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab
], dtype=float)

# --------------------------------------------------------------
# 1. PBPK 方程 & 模拟函数 —— 直接引用 modfit0428 版本:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
#    （若您的 modfit 已更新，只需更新 import）
# --------------------------------------------------------------
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    input_rate = R if t <= T_total else 0
    dy = np.zeros(7)
    dy[0] = (QRest*y[3]/VRest/PRest + QK*y[2]/VK/PK + QL*y[1]/VL/PL
             - QPlas*y[0]/VPlas + Kreab*y[4] + input_rate/VPlas)
    dy[1] = QL*(y[0]/VPlas - y[1]/VL/PL) - Kbile*y[1]
    dy[2] = QK*(y[0]/VPlas - y[2]/VK/PK) - y[0]/VPlas*GFR*Free \
            - (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK)
    dy[3] = QRest*(y[0]/VPlas - y[3]/VRest/PRest)
    dy[4] = y[0]/VPlas*GFR*Free + (Vmax_baso*y[2]/VK/PK)/(Km_baso + y[2]/VK/PK) \
            - y[4]*Kurine - Kreab*y[4]
    dy[5] = Kurine*y[4]
    dy[6] = Kbile*y[1]
    return dy

def FIT_model(t, D_total, T_total, *params):
    """返回血浆浓度–时间曲线 (mg/L)"""
    R  = D_total / T_total              # 恒速滴注 (mg·h⁻¹)
    y0 = np.zeros(7)
    sol = odeint(derivshiv, y0, t,
                 args=(params, R, T_total),
                 rtol=1e-6, atol=1e-9, h0=0.1)
    return sol[:, 0] / VPlas            # 仅血浆室浓度

# --------------------------------------------------------------
# 2. Step-1 —— 全局灵敏度 (胡-式 RMS) 排序
# --------------------------------------------------------------
del_rel = 0.01                                # 相对扰动 1 %
s_vectors = []                              # 储存 sl(t) 向量
S_global  = []                              # RMS 敏感度指标

# ★ 先算“基线输出”并拼接成一条长向量（77 组拼一起）
y_base_all = []
for t_pts, D, Tin in zip(time_groups, dose_groups, tinf_groups):
    y_base_all.append(FIT_model(t_pts, D, Tin, *baseline))
y_base_all = np.concatenate(y_base_all)
xit_y = np.std(y_base_all)                     # 用于无量纲化

for idx, (pname, n0) in enumerate(zip(param_names, baseline)):
    deln = n0*del_rel if n0 != 0 else 1e-6      # 绝对扰动
    up   = baseline.copy(); up[idx] += deln
    down = baseline.copy(); down[idx] -= deln

    diff_concat = []                        # 存储该参数在全部病人上的差分
    for t_pts, D, Tin in zip(time_groups, dose_groups, tinf_groups):
        y_up   = FIT_model(t_pts, D, Tin, *up)
        y_down = FIT_model(t_pts, D, Tin, *down)
        diff_concat.append((y_up - y_down)/(2*deln))

    sl_vec = (deln/xit_y) * np.concatenate(diff_concat)  # 公式 (3)
    s_vectors.append(sl_vec)
    S_global.append(math.sqrt(np.mean(sl_vec**2)))

# 排序并打印
order = np.argsort(S_global)[::-1]
print("\n=== Step-1  全局灵敏度排序 (高→低) ===")
for rk, idx in enumerate(order, 1):
    print(f"{rk:2}. {param_names[idx]:<12}  S = {S_global[idx]:.3e}")

plt.figure(figsize=(8,4))
plt.bar([param_names[i] for i in order], [S_global[i] for i in order])
plt.ylabel("Global sensitivity $S_l$")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("sens_ranking.png", dpi=300)

# --------------------------------------------------------------
# 3. Step-2 —— 共线性指数 Y (子集遍历)
# --------------------------------------------------------------
unit_vecs = [v/np.linalg.norm(v) for v in s_vectors]

def gamma(indices):
    """Y = 1 / √lam_min(ŜᵀŜ) (文献公式 8)"""
    S = np.column_stack([unit_vecs[i] for i in indices])
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)
    return 1/math.sqrt(lam_min)

sizes, Y_low, Y_high = [], [], []
for k in range(1, len(param_names)+1):
    Yvals = [gamma(comb) for comb in itertools.combinations(range(len(param_names)), k)]
    sizes.append(k)
    Y_low.append(min(Yvals))
    Y_high.append(max(Yvals))

plt.figure()
plt.fill_between(sizes, Y_low, Y_high, alpha=.3, label="Y range")
plt.plot(sizes, Y_high, "o-", lw=1)
plt.axhline(10, ls='--', c='r'); plt.axhline(15, ls='--', c='r')
plt.xlabel("size of parameter subset")
plt.ylabel("collinearity index Y")
plt.tight_layout()
plt.savefig("gamma_ranges.png", dpi=300)

# --------------------------------------------------------------
# 4. Step-3 —— Y 最大的 2-元 & 3-元子集
# --------------------------------------------------------------
def worst(k, top=5):
    combs = itertools.combinations(range(len(param_names)), k)
    ranked = sorted(((gamma(c), c) for c in combs), key=lambda x: -x[0])
    return [(g, [param_names[i] for i in idxs]) for g, idxs in ranked[:top]]

for k in (2, 3):
    print(f"\n=== Step-3  Y 最高的 {k}-元子集 (前{len(worst(k))}) ===")
    for g, subset in worst(k):
        print(f"Y = {g:6.2f}  ->  {subset}")

print("\n分析完成！图形见 sens_ranking.png 与 gamma_ranges.png")

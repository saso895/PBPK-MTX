# -*- coding: utf-8 -*-
"""
lake_sens02_plus.py —— 在一次敏感性+共线性分析基础上，
                      计算 Table-8 风格的参数标准误差 & 相关系数
-------------------------------------------------------------------
新增 Step-5:
  · 选定一个参数子集 (默认取全局灵敏度最高的 7 个)
  · 构造“加权雅可比” J_w  (与 WSS 中的加权方式一致)
  · 用 (J_w^T J_w)^-1 估计协方差矩阵，给出:
      - 相对标准误差 (%)
      - 参数间相关系数 ρ_jk
  · 结果打印 + 写入 table8_stats.csv
-------------------------------------------------------------------
其余 Step-1～Step-4 与原文件完全一致。
"""

import itertools, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg as la                     # 解决 la
from scipy.optimize import least_squares      # 解决 least_squares
from numpy.random import default_rng  

# --------------------------------------------------------------
# 0. 载入固定/可调参数 & 病人给药数据
# --------------------------------------------------------------
from init_param import (
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
    PRest, PK, PL, Kbile, GFR, Free,
    Vmax_baso, Km_baso, Kurine, Kreab
)
from init_data_point4 import (
    time_points_train as time_groups,
    input_dose_train  as dose_groups,
    inject_timelen_train as tinf_groups,
    concentration_data_train as conc_groups,
)

param_names = [
    "PRest", "PK", "PL", "Kbile", "GFR",
    "Free",  "Vmax_baso", "Km_baso", "Kurine", "Kreab",
]
baseline = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
], dtype=float)

# --------------------------------------------------------------
# 1. PBPK 方程 & 模拟函数
# --------------------------------------------------------------
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    inp = R if t <= T_total else 0
    dy  = np.zeros(7)
    dy[0] = (
        QRest * y[3] / VRest / PRest
        + QK * y[2] / VK / PK
        + QL * y[1] / VL / PL
        - QPlas * y[0] / VPlas
        + Kreab * y[4]
        + inp / VPlas
    )
    dy[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    dy[2] = (
        QK * (y[0] / VPlas - y[2] / VK / PK)
        - y[0] / VPlas * GFR * Free
        - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    )
    dy[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    dy[4] = (
        y[0] / VPlas * GFR * Free
        + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
        - y[4] * Kurine
        - Kreab * y[4]
    )
    dy[5] = Kurine * y[4]
    dy[6] = Kbile * y[1]
    return dy

def FIT_model(t, dose, tinf, *params):
    R = dose / tinf
    y0 = np.zeros(7)
    sol = odeint(
        derivshiv, y0, t,
        args=(params, R, tinf),
        rtol=1e-6, atol=1e-9, h0=0.1,
    )
    return sol[:, 0] / VPlas   # 血浆浓度

# --------------------------------------------------------------
# 2. Step-1 —— 全局灵敏度
#使用文章中公式4-6计算
# --------------------------------------------------------------
delta_rel = 0.01
sl_vectors, S_global = [], []

base_curve, weights = [], []
for t, d, tinf in zip(time_groups, dose_groups, tinf_groups):
    y = FIT_model(t, d, tinf, *baseline)
    base_curve.extend(y)         #原始参数拟合浓度拼接
    w = np.ones_like(y);         # 第二点权重×2（示例）
    if len(y) >= 2: w[1] = 2.0
    weights.extend(w)

base_curve = np.array(base_curve)
weights    = np.array(weights)
sy = np.sqrt(np.average(
        (base_curve - np.average(base_curve, weights=weights))**2,
        weights=weights))

for idx, (pname, theta0) in enumerate(zip(param_names, baseline)):
    dtheta = theta0 * delta_rel if theta0 != 0 else 1e-6
    up, down = baseline.copy(), baseline.copy()
    up[idx] += dtheta; down[idx] -= dtheta
    diff = np.concatenate([
        FIT_model(t, d, tinf, *up) - FIT_model(t, d, tinf, *down)
        for t, d, tinf in zip(time_groups, dose_groups, tinf_groups)
    ]) / (2*dtheta)
    sl = (dtheta/sy) * (diff * weights)
    sl_vectors.append(sl)
    S_global.append(math.sqrt(np.mean(sl**2)))

order = np.argsort(S_global)[::-1]
print("\n=== Step-1  全局灵敏度排序 ===")
for rk, idx in enumerate(order, 1):
    print(f"{rk:2}. {param_names[idx]:<12}  S = {S_global[idx]:.3e}")

# --------------------------------------------------------------
# 3. Step-2 —— 共线性 γ（与你原脚本相同02）
# --------------------------------------------------------------
'''np是上面导入的 numpy
linalg.numpy.linalg是 numpy 里的“线性代数”模块
norm(v)：就是计算向量 v 的欧几里得范数（= 其长度/模长）
例如v = [3, 4]，则 np.linalg.norm(v) 结果是 5
最终 unit_vecs：就是“每个参数的灵敏度单位向量”，每一项都是个长度为1的 numpy 数组'''
unit_vecs = [v/np.linalg.norm(v) for v in sl_vectors]#公式7

'''
idxs对应0-9,10个参数编号
S = np.column_stack([unit_vecs[i] for i in idxs])
unit_vecs[i]：挑出第 i 个参数的单位灵敏度向量，[unit_vecs[i] for i in idxs]：组成一个向量列表
np.column_stack(...)：把这些向量“按列”拼成一个二维矩阵
比如有 3 个参数，每个单位向量长度 10，拼成的 S 是 10 行 3 列的矩阵，每列是一个参数的作用方向
lam_min = np.min(np.linalg.eigvals(S.T @ S).real)
S.T：S 的转置（行和列交换），S.T @ S：先转置再点乘自己，得到一个“方阵”（即 𝑆𝑇STS）
np.linalg.eigvals(...)：计算这个方阵的所有特征值（eigenvalue）这在数学里衡量“这些向量有多独立”
.real：只取实部（因为数值运算里可能有虚数，但我们只关心实数部分）
np.min(...)：找出最小的特征值
return 1 / math.sqrt(lam_min)，math.sqrt()：开平方（square root）
1 / sqrt(lam_min)：就是文献里的共线性指数 γ
'''
def gamma(idxs):
    S = np.column_stack([unit_vecs[i] for i in idxs])#公式7的小s构造出大S
    lam_min = np.min(np.linalg.eigvals(S.T @ S).real)#公式8根号里的内容
    return 1/math.sqrt(lam_min)#公式8最终结果

# γ 包络 (min/max) 
'''
size_list：用于保存每一次循环参数组合的“子集大小 k”
g_low：保存每种 k 下，所有组合的 γ 的最小值（共线性最弱的情况）
g_high：保存每种 k 下，所有组合的 γ 的最大值（共线性最强的情况）
len(param_names)：参数总数
range(1, len(param_names) + 1)：
从 1 开始（即1个参数），到参数总数（包含所有参数的组合）
for k in ...：对每种参数子集大小（k个参数组合）都循环一次
range(len(param_names))：生成 [0, 1, 2, ..., n-1] 的整数序列，n是参数数
itertools.combinations(..., k)：生成所有从 n 个参数里任选 k 个的全部不重复组合
举例：如果有 3 个参数，k=2，那么组合就是 (0,1), (0,2), (1,2)
c：每个 c 就是当前要分析的一组参数的“下标”（比如 c = (0,2,3)）
[gamma(c) for c in ...]：
对每一个组合 c，调用你前面定义的 gamma(c) 函数
计算出每一组参数的共线性指数 γ
最终 g_vals 是一个列表，里面存的是本轮所有 k 元组的 γ 值
'''
#size_list, g_low, g_high = [], [], []
plt.figure()
for k in range(1, len(param_names) + 1):
    g_vals = [gamma(c) for c in itertools.combinations(range(len(param_names)), k)]
    #size_list.append(k)
    #g_low.append(min(g_vals)); g_high.append(max(g_vals))
    xvals = [k] * len(g_vals)
    plt.plot(xvals, g_vals, 'ko', alpha=0.6, markersize=4)  # 黑色圆点
# 画斜线（如要标注灵敏度Top-n子集的γ）
# 例：用灵敏度最高7个参数顺次累计形成的子集
special_idxs = order[:10]
special_gamma = []
for k in range(1, len(special_idxs)+1):
    idxs = tuple(special_idxs[:k])
    g = gamma(idxs)
    plt.plot(k, g, 'rs', markersize=7)  # 红色方块标记
    special_gamma.append(g)
plt.plot(range(1, len(special_gamma)+1), special_gamma, 'r-', lw=2, label='Top-n subset γ')

# 横线标注
plt.axhline(10, ls="--", c="r", lw=1, label='γ=10')
plt.axhline(15, ls="--", c="r", lw=1, label='γ=15')

plt.xlabel("Subset size (k)")
plt.ylabel("Collinearity index γ")
plt.title("Collinearity indices for all parameter subsets")
plt.legend()
plt.tight_layout()
plt.savefig("gamma_verticals02.png", dpi=300)
plt.show()

# --------------------------------------------------------------
# 4. Step-3 —— 所有 γ 写 Excel (保持原先版本)
# --------------------------------------------------------------
rows = []
for k in range(1, len(param_names) + 1):
    for idxs in itertools.combinations(range(len(param_names)), k):
        g = gamma(idxs)
        subset_names = ", ".join(param_names[i] for i in idxs)
        rows.append({"Subset size": k, "Parameters": subset_names, "Gamma": g})

# 转 DataFrame
GammaDF = pd.DataFrame(rows)

# 按子集大小分工作表写入
with pd.ExcelWriter("gamma_subsets01.xlsx", engine="openpyxl") as writer:
    for k, grp in GammaDF.groupby("Subset size"):
        grp_sorted = grp.sort_values("Gamma", ascending=False)
        grp_sorted.to_excel(writer, sheet_name=f"size_{k}", index=False)
print("已生成 gamma_subsets.xlsx —— 请在本目录查看所有子集 γ 值")

# --------------------------------------------------------------
# 5. 追加 Step-5 —— 近似标准误差 & 相关系数 (Table 8)
# --------------------------------------------------------------
print("\n=== Step-5  近似标准误差 & 相关系数 (Table-8) ===")

# —— 5.1 选“要估计的子集”：可指定共线性分析结果最优组合 ——
# 手动指定最优共线性组合（例：γ=5.22, 包含 Kbile 且敏感性排名前5）
#chosen_subset_names = [
#    "PRest", "Kbile", "GFR", "Free", "Vmax_baso", "Km_baso", "Kurine", "Kreab"
#]
chosen_subset_names =param_names    #待优化子集
# 自动转成下标
chosen_subset_idx = [param_names.index(p) for p in chosen_subset_names]  #记录子集下标
subset_idx = chosen_subset_idx  #子集下标
subset_names = chosen_subset_names #子集名称
# 如需恢复用Top-N灵敏度参数，只需把上面两行注释掉，改回:
# subset_idx = order[:7]
# subset_names = [param_names[i] for i in subset_idx]

p = len(subset_idx) #子集长度
print("参数子集:", subset_names)

# ============================================================
# 6. WSS 目标函数 + 最小二乘拟合
# ============================================================
print("\n=== Step‑6  WSS 目标函数 + 参数拟合优化 ===")
# ------------------------------------------------------------
#    ──把“每个人第 2 个浓度点”权重加倍
# ------------------------------------------------------------
sc_groups = []
for obs in conc_groups:                # obs 是一条实验的观测浓度数组
    sc = np.ones_like(obs)             # 先全部权重 = 1
    if len(sc) > 1:                    # 若至少有 2 个采样点
        sc[1] = 0.1                    # 让第 2 点的 sc 更小 → 权重更大
    sc_groups.append(sc)               # (权重 ∝ 1/sc)
# ------------------------------------------------------------
#    计算归一化残差
# ------------------------------------------------------------
def _residuals(theta_sub):        # theta_sub 是待估 n 个参数的当前试探值
    full = baseline.copy()        # 复制一份全 10 维参数向量
    full[subset_idx] = theta_sub  # 用试探值替换 n 维敏感参数子集
    res = []                      # 准备累计所有实验的残差
    for t, d, tinf, obs,sc in zip(   # 同时遍历每条给药实验：
            time_groups, dose_groups, #  ├─ t     → 采样时间点数组
            tinf_groups, conc_groups,sc_groups):#  ├─ d,tinf→ 剂量与输注时长
        res.extend((FIT_model(t, d, tinf, *full) #     预测浓度曲线
                   - obs)/sc)             #   − 实测浓度 → 残差
    return np.asarray(res)            # 返回 1-D 残差向量 (拼接所有实验)

lb = np.zeros(len(subset_idx))        # 各参数下界 0（不可负）
ub = np.full(len(subset_idx), np.inf) # 上界默认为 +∞
ub[subset_names.index("Free")] = 1.0  # 生理约束 Free≤1

# ------------------------------------------------------------
#    计算归一化残差
# ------------------------------------------------------------
#----------在 SciPy 的 least_squares 框架里，“平方 + 求和”这一动作是由求解器自己完成的
'''
把当前参数 θ 送进 _residuals ⇒ 得到向量 r。
计算 cost = ½ ||r||² = ½ ∑𝑟𝑘2​ 。
用数值雅可比更新 θ，循环 1–2 直至收敛
'''
opt = least_squares(                  #   Levenberg-Marquardt/Trust-Region 拟合
    _residuals,                       #   目标函数：加权残差
    baseline[subset_idx],             #   初值：基线生理参数
    bounds=(lb, ub),                  #   简单界限约束
    method="trf",                     #   使用 trust-region reflective 算法
    xtol=1e-10,                       #   参数步长 (step norm) 阈值
    verbose=2,                        #   输出迭代信息
    max_nfev=300)                     #   最多 300 次函数评估
theta_hat = opt.x                     #   最终最小二乘估计值，x就是使拟合误差最小的参数值。
rss = np.sum(opt.fun**2)              #   残差平方和 (cost×2)
dof = len(opt.fun) - len(theta_hat)   #   自由度 = 数据点数 − 参数数
print(f"优化完成  RSS = {rss:.4g}  (DOF = {dof})")
print(f"success : {opt.success}")        # True / False
print(f"status  : {opt.status}")         # 0–5 的代码，见 SciPy 文档
print(f"message : {opt.message}")        # 人可读的收敛说明
print(f"nfev    : {opt.nfev}")           # 目标函数评估次数
baseline[subset_idx] = theta_hat  # 回写以便继续迭代

# ============================================================
# 7. 协方差 & 不确定度传播 / 验证 公式（10）
# ============================================================
print("\n=== Step‑7  协方差 + 验证 / 预测 ===")

J = opt.jac                      # ① 最终迭代点的雅可比矩阵 (∂r/∂θ),	SciPy least_squares 保存的雅可比矩阵 
sigma2 = rss / dof               # ② 残差方差估计 σ² = RSS / 自由度
cov = sigma2 * la.inv(J.T @ J)   # ③ 线性近似协方差：σ²·(JᵀJ)⁻¹
se  = np.sqrt(np.diag(cov))      # ④ 每个参数的标准误差 = 协方差对角元素开方

fit_tbl = pd.DataFrame({         # ⑤ 汇总成表便于查看 & 导出
    "Parameter": subset_names,
    "Estimate":  theta_hat,
    "StdErr":    se,
    "RelSE_%":   100 * se / np.abs(theta_hat),
})

# -- 7.1 拟合曲线示例 --
print("绘制示例拟合曲线 …")
idx0 = 0
_t, _d, _ti, _obs = (time_groups[idx0], dose_groups[idx0],
                     tinf_groups[idx0], conc_groups[idx0])
_pred = FIT_model(_t, _d, _ti, *baseline)
plt.figure(); plt.plot(_t, _obs, "o", label="Obs"); plt.plot(_t, _pred, "-", label="Fit");
plt.xlabel("Time (h)"); plt.ylabel("Plasma conc. (mg/L)"); plt.legend(); plt.tight_layout();
plt.savefig("fit_curve_example.png", dpi=300); plt.close()

# -- 7.2 Monte‑Carlo 95 % 预测带 --
print("Monte‑Carlo 采样生成预测带 …")
_rng = default_rng(2025)
N_MC = 300
samps = _rng.multivariate_normal(theta_hat, cov, size=N_MC)

def _fill_params(s):
    tmp = baseline.copy(); tmp[subset_idx] = s; return tmp
pred_mat = np.array([FIT_model(_t, _d, _ti, *_fill_params(s)) for s in samps])
mean_curve = pred_mat.mean(axis=0)
pi_low, pi_high = np.percentile(pred_mat, [2.5, 97.5], axis=0)
plt.figure(); plt.fill_between(_t, pi_low, pi_high, alpha=0.3, label="95% PI");
plt.plot(_t, _obs, "o", label="Obs"); plt.plot(_t, mean_curve, "-", label="Pred (mean)");
plt.xlabel("Time (h)"); plt.ylabel("Plasma conc. (mg/L)"); plt.legend(); plt.tight_layout();
plt.savefig("fit_curve_PI.png", dpi=300); plt.close()

# -- 7.3 保存结果 --
fit_tbl.to_csv("step6_fitted_params.csv", index=False)
print("参数表已写入 step6_fitted_params.csv")
print("示例曲线: fit_curve_example.png")
print("95% 预测带: fit_curve_PI.png")

if __name__ == "__main__":
    print(fit_tbl)

# simulate.py  ------------------------------------------------------------
import pandas as pd          # ← NEW
import numpy as np           # ← NEW
import matplotlib.pyplot as plt
from model_mini import simulate
from tqdm import tqdm 
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
def simulate_progress(dose, route="iv", t_end=24, step_h=1.0,
                      rtol=1e-4, atol=1e-6):
    """
    和原来 simulate() 返回一样 (t, Cp, sol)，但会实时刷新进度条。
      dose     — mg
      route    — "iv" or "po"
      t_end    — 模拟到多少小时
      step_h   — 每 step_h 小时做一次 solve_ivp，然后更新进度
    """
    # ① 首先把 model_mini 里的 simulate 拿来用，得到 y0 & 参数
    from model_mini import pbpk_ode, _init_state   # 只在这一行引用
    y0, p = _init_state(dose, route)               # _init_state 是你 model_mini 里
                                                   # 初始化 y0 参数的私有函数

    # ② 循环分段积分
    t_all, Cp_all = [], []
    with tqdm(total=t_end, desc=f"{dose} mg {route}", unit="h") as bar:
        t0, y_now = 0.0, y0
        while t0 < t_end:
            t1 = min(t0 + step_h, t_end)
            sol = solve_ivp(pbpk_ode, [t0, t1], y_now,
                            args=(route, p),
                            method="LSODA",
                            rtol=rtol, atol=atol)
            # 记录结果
            t_all.extend(sol.t[1:])        # 不重复第 1 点
            Cp_all.extend(sol.y[0,1:])     # 假设第 0 条是血浆 MTX
            # 更新 “初值” 为当前区段末端
            y_now = sol.y[:,-1]
            t0 = t1
            bar.update(t1 - bar.n)         # 刷新进度条

    t_arr = np.array(t_all)
    Cp_arr = np.array(Cp_all)
    return t_arr, Cp_arr, None             # 第 3 个返回值可随意
# -----------------------------------------------------------------------------


# 1) 15 mg IV bolus
time = 24
t1, Cp1, _ = simulate_progress(15, route="iv", t_end=time)

# 2) 25 mg oral
#t2, Cp2, _ = simulate(25, route="po", t_end=time)
# ── 读入图 3a 的观测点 ───────────────────────────────────────────
obs = pd.read_csv("mini_figure3c.csv")   # CSV 放在同目录
t_obs  = obs["Time"].to_numpy()        # 列名按你的 CSV 而定
Cp_obs = obs["Conc"].to_numpy()     # 〃


plt.figure(figsize=(2,6))
plt.semilogy(t_obs, Cp_obs, "ko", mfc="none", label="Observed 15 mg IV")  # 新增观察点
plt.semilogy(t1, Cp1, label="PBPK 15 mg IV")                              # 原来就有
# plt.semilogy(t2, Cp2, label="PBPK 25 mg PO")                            # 若无关可先注释

plt.xlabel("Time (h)")
plt.ylabel("Plasma MTX (mg/L)")
plt.ylim(1e-4, 1e1)                    # 纵轴范围可按图 3a 调
plt.legend()
plt.tight_layout()
plt.show()
print(f"\n🌟 已运行完毕")



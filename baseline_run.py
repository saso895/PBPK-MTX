import numpy as np, pathlib, datetime, subprocess, json
from init_param import (
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab, VPlas
)
from ode_core import derivshiv          # ← 复用你已有的 derivshiv
from checks import mass_balance, write_report
from scipy.integrate import odeint
# ------------------------------------------------------------
baseline_init = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
], dtype=float)

OUT = pathlib.Path("saved_result")
OUT.mkdir(exist_ok=True)

# ---------- 1) 一次完整模拟 （拿到 sol 7 列） ---------------
t = np.linspace(0, 120, 1201)
dose_mg = 1000
R = dose_mg / 0.5
y0 = np.zeros(7)
sol = odeint(derivshiv, y0, t,
             args=(baseline_init, R, 0.5),
             rtol=1e-6, atol=1e-9)

conc = sol[:, 0] / VPlas                # 仍输出血浆浓度
np.savetxt(OUT/"conc_pred.csv",
           np.column_stack([t, conc]),
           header="time_h,conc_mg/L", delimiter=",")

# ---------- 2) 质量平衡 -------------------------------------
rec = mass_balance(sol, dose_mg)
write_report(rec, OUT)

print(f"✓ baseline 完成，Recovery={rec:.4f}")


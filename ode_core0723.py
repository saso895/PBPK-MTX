
import itertools, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg as la                     # 解决 la
from scipy.optimize import least_squares      # 解决 least_squares
from numpy.random import default_rng  
import numpy as np 
import os
from tqdm import tqdm
import pickle
import datetime

today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# --------------------------------------------------------------
# 0. 载入固定/可调参数 & 病人给药数据
# --------------------------------------------------------------
from init_param0723 import (
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
    PRest, PK, PL, Kbile, GFR, Free,
    Vmax_baso, Km_baso, Kurine, Kreab,
    # === MOD BEGIN 2025‑07‑23 新增参数引入 ===
    Vmax_apical, Km_apical, Vmax_bile, Km_bile
    # === MOD END ============================
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
    # === MOD BEGIN 2025‑07‑23 新增参数名 ===
    "Vmax_apical", "Km_apical", "Vmax_bile", "Km_bile"
    # === MOD END ===========================
]
baseline_init = np.array([
    PRest, PK, PL, Kbile, GFR,
    Free,  Vmax_baso, Km_baso, Kurine, Kreab,
        # === MOD BEGIN 2025‑07‑23 新增参数基线 ===
    Vmax_apical, Km_apical, Vmax_bile, Km_bile
    # === MOD END ===========================
], dtype=float)

# ---------- 对数 <-> 线性 ----------
def log_normalize(p):          # 取对数
    return np.log(np.asarray(p, dtype=float))

def exp_denormalize(lp):       # 反对数
    return np.exp(np.asarray(lp, dtype=float))


# --------------------------------------------------------------
# 1. PBPK 微分方程
# --------------------------------------------------------------
def derivshiv(y, t, parms, R, T_total):
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab,Vmax_apical, Km_apical, Vmax_bile, Km_bile= parms
    inp = R if t <= T_total else 0
    dy  = np.zeros(7)

        # —— 计算各组织浓度 —— 
    C_plas  = y[0] / VPlas
    C_liver = y[1] / VL / PL
    C_kid   = y[2] / VK / PK

    # —— 非线性转运速率（米氏方程） ——
    baso_uptake   = (Vmax_baso   * C_kid)   / (Km_baso   + C_kid)        # 血→肾细胞
    apical_efflux = (Vmax_apical * C_kid)   / (Km_apical + C_kid)        # 肾细胞→管腔
    bile_secr_sat = (Vmax_bile   * C_liver) / (Km_bile   + C_liver)      # 肝细胞→胆汁
    bile_secr_lin = Kbile * y[1]                                             # 保留原线性项
    bile_secr     = bile_secr_lin + bile_secr_sat

    # —— 微分方程 —— --------------------------------------------
    # [0] Plasma
    dy[0] = (
        QRest * y[3] / VRest / PRest
        + QK * y[2] / VK / PK
        + QL * y[1] / VL / PL
        - QPlas * C_plas
        + Kreab * y[4]
        + inp 
    )
    # [1] Liver
    dy[1] = QL * (C_plas - C_liver) - bile_secr
    # [2] Kidney tissue
    dy[2] = (
        QK * (C_plas - C_kid)
        - C_plas * GFR * Free
        - baso_uptake
        - apical_efflux        
    )
    # [3] Rest of body
    dy[3] = QRest * (C_plas - y[3] / VRest / PRest)
    # [4] Tubule lumen
    dy[4] = (
        C_plas * GFR * Free
        + baso_uptake
        + apical_efflux        
        - y[4] * Kurine
        - Kreab * y[4]
    )
    # [5] Urine
    dy[5] = Kurine * y[4]
    # [6] Bile
    dy[6] = bile_secr
    return dy
# --------------------------------------------------------------
# 2.模拟函数（离散）
# --------------------------------------------------------------
def FIT_model(t, dose, tinf, *params):
    R = dose / tinf
    y0 = np.zeros(7)
    sol = odeint(
        derivshiv, y0, t,
        args=(params, R, tinf),
        rtol=1e-6, 
        atol=1e-9, 
        h0=0.1,
        mxstep=10000
    )
    return sol[:, 0] / VPlas   # 血浆浓度
# --------------------------------------------------------------
# 2.模拟函数（连续）
# --------------------------------------------------------------
def PK_model(t, D_total, T_total, Duration,*param):
    '''药代动力学模型函数，用于参数拟合'''
    #print(f"params : {param}") 
    # 计算注射速率
    R = D_total / T_total
    y0 = np.zeros(7)#(10e-6)+
    # Specify time points to simulate
    Time=np.arange(0, Duration + 0.1, 0.1)
    # 调用 odeint 进行数值积分，传入微分方程 derivshiv 和初始条件 y0
    y = odeint(
        derivshiv, 
        y0, 
        Time, 
        args=(param, R, T_total), 
        #method='BDF',
        rtol=1e-4,  # 放宽相对误差容忍度
        atol=1e-7,  # 放宽绝对误差容忍度
        h0=1e-5     # 设置初始步长
    )
    # y = odeint(derivshiv, y0, t, args=(params, R, T_total), rtol=1e-5, atol=1e-8)    
    #return y[:, 0] / VPlas
    CA = y[:, 0] / VPlas
    results = np.column_stack((Time, CA))
    return results
# -*- coding: utf-8 -*-
"""
Genetic‑Algorithm parameter fitting for the PBPK model.

This script re‑uses the ODE system and data‑loading utilities that are
already present in the MCMC example (``test0409.py``).  The model
structure is unchanged – only the optimisation strategy has been
switched to a single‑objective genetic algorithm implemented with
GEATpy.  The objective is the total root‑mean‑square error (RMSE)
between the simulated plasma concentration curve and all training
measurements.

Usage
-----
$ python pbpk_ga_fit.py [--seed 42] [--ngen 200] [--pop 100]

After completion the script writes two artefacts under
``saved_result/``:
1. ``ga_best_params.npy`` – best parameter vector (shape ``(10,)``)
2. ``ga_convergence.csv`` – generation‑wise best / mean fitness

Dependencies
------------
* GEATpy ≥ 2.4.0  (https://github.com/geatpy-dev/geatpy)
* NumPy, pandas, scipy, tqdm, matplotlib (optional for quick plot)
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pickle
# ---- domain‑specific imports from the existing project -------------
from init_param import (
    QRest, QK, QL, QPlas, VRest, VK, VL, VPlas,
)
from init_data_point4 import(
    time_points_train,
    concentration_data_train,
    input_dose_train,
    inject_timelen_train,
)

# Optional: use previously optimised (e.g. modfit) params as a
# reasonable centre of the search space if available.
load_saved_data=False
if load_saved_data:
    #_init_guess = np.load("saved_result/optimized_params.npy")
    with open('optimized_params0427_Powell.pkl', 'rb') as f:#0506修改
        _init_guess = pickle.load(f)
else:
    import init_param as ip
    # from init_param import PRest,PK,PL,Kbile,GFR,Free,Vmax_baso,Km_baso,Kurine,Kreab
    _init_guess = np.array([ip.PRest, ip.PK, ip.PL, ip.Kbile, ip.GFR, ip.Free, ip.Vmax_baso, ip.Km_baso, ip.Kurine, ip.Kreab])

###############################################################################
# 1. PBPK ODE system (copied verbatim from the reference script)              #
###############################################################################

def derivshiv(y, t, parms, R, T_total):
    '''定义微分方程的函数，包含药物点滴输入'''
    
    PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab = parms
    # 确保 input_rate 是标量
    input_rate = R if t <= T_total else 0
    ydot = np.zeros(7)
    ydot[0] = (QRest * y[3] / VRest / PRest) + (QK * y[2] / VK / PK) + (QL * y[1] / VL / PL) - (QPlas * y[0] / VPlas) + Kreab * y[4] + input_rate / VPlas
    ydot[1] = QL * (y[0] / VPlas - y[1] / VL / PL) - Kbile * y[1]
    ydot[2] = QK * (y[0] / VPlas - y[2] / VK / PK) - y[0] / VPlas * GFR * Free - (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK)
    ydot[3] = QRest * (y[0] / VPlas - y[3] / VRest / PRest)
    ydot[4] = y[0] / VPlas * GFR * Free + (Vmax_baso * y[2] / VK / PK) / (Km_baso + y[2] / VK / PK) - y[4] * Kurine - Kreab * y[4]
    ydot[5] = Kurine * y[4]
    ydot[6] = Kbile * y[1]
    return ydot

def pk_model(Time, D_total, T_total, params):
    """Return plasma concentration (VPlas normalised) at the given time points."""
    R = D_total / T_total
    y0 = np.zeros(7)
    sol = odeint(
        derivshiv,
        y0,
        Time,
        args=(params, R, T_total),
        method='BDF',#0506添加
        rtol=1e-4,
        atol=1e-7,
    )
    return sol[:, 0] / VPlas

###############################################################################
# 2. Fitness (objective) function                                             #
###############################################################################

def rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))


def fitness_func(params):
    """Total RMSE across all subjects in the training data."""
    total_err = 0.0
    for t, conc, dose, tlen in zip(
        time_points_train,
        concentration_data_train,
        input_dose_train,
        inject_timelen_train,
    ):
        pred = pk_model(t, dose, tlen, params)
        # small epsilon to avoid log(0) downstream (if any)
        total_err += rmse(pred, conc + 1e-6)
    return total_err

###############################################################################
# 3. Genetic algorithm with GEATpy                                            #
###############################################################################

# Import GEATpy only when needed; this avoids ImportError if absent.
try:
    import geatpy as ea
except ImportError as ex:
    raise ImportError(
        "GEATpy is required for this script.  Install via 'pip install geatpy'."
    ) from ex

# ---- Parameter bounds -------------------------------------------------------
# Here we use log‑scale bounds spanning two orders of magnitude around the
# initial guess.  Adjust as necessary for your specific PBPK model.

_lb = _init_guess * 0.1
_ub = _init_guess * 10.0

# Ensure Free (fraction unbound 0‑1) stays within [0,1].
_lb[5] = 1e-4
_ub[5] = 1.0

###############################################################################
# 4. GEATpy problem definition                                                #
###############################################################################

class PBPKProblem(ea.Problem):
    def __init__(self):
        name = "PBPK_GA_Fit"
        M = 1                     # 目标数
        Dim = 10                  # 决策变量维度
        maxormins = [1] * M       # 1 → 最小化
        varTypes = [0] * Dim      # 连续型

        lb, ub = _lb, _ub
        lbin = [1] * Dim
        ubin = [1] * Dim

        # 注意参数顺序：name, M, maxormins, Dim, ...
        super().__init__(name, M, maxormins, Dim,
                         varTypes, lb, ub, lbin=lbin, ubin=ubin)
        # name, M, maxormins, Dim, varTypes, lb, ub, lbin=None, ubin=None, aimFunc=None, evalVars=None, calReferObjV=None

    def aimFunc(self, pop):
        # Evaluate the objective for each individual.
        vars_pop = pop.Phen  # shape (pop_size, dim)
        obj_values = np.zeros((vars_pop.shape[0], 1))
        for i, indiv in enumerate(vars_pop):
            obj_values[i, 0] = fitness_func(indiv)
        pop.ObjV = obj_values

###############################################################################
# 5. Main optimisation loop                                                   #
###############################################################################

def main(args):
    problem = PBPKProblem()
    # ── 生成 Field 描述符（一次即可复用） ──────────────────────────────
    Dim     = problem.Dim
    ranges  = np.vstack((_lb, _ub))           # 2×Dim，下界在第一行
    borders = np.ones((2, Dim), dtype=int)    # 全闭区间［lb, ub］
    encoding = 'RI'
    varTypes = [0]*Dim

    # ── 2) 创建 Field 描述符 ───────────────────────────────────────────
    Field = ea.crtfld(encoding, problem.varTypes, ranges, borders)

    # ── 3) 按新版 API 创建种群 ─────────────────────────────────────────
    population = ea.Population(Encoding=encoding, NIND=args.pop, Field=Field)
    # Algorithm: simple GA (also works: soea_DE_rand_1_bin)
    algorithm = ea.soea_EGA_templet(
        problem,
        population,
        MAXGEN=args.ngen,
        logTras=1,
        trappedValue=1e-6,
        maxTrappedCount=50,
        drawing=1,  # 0: no real‑time plotting (makes remote runs faster)
    )

    # Stochastic operators configuration – tweak as desired
    algorithm.mutOper.Pm = 1 / problem.Dim

    # Run optimisation
    res = ea.optimize(algorithm, precision=[1e-6] * problem.Dim, seed=args.seed, verbose=True)
    
    if res['success']:
        best_param = res['Vars']
        os.makedirs("saved_result", exist_ok=True)
        np.save("saved_result/ga_best_params.npy", best_param)

        # Save convergence
        # pd.DataFrame({
        #     "gen": np.arange(len(res[5])) + 1,
        #     "best": res[5],
        #     "mean": res[6],
        # }).to_csv("saved_result/ga_convergence.csv", index=False)

        print("\nBest parameters (GA):\n", best_param)
        # print("Fitness (total RMSE):", res[0])
    else:
        print('没找到可行解。')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ngen", type=int, default=200, help="Number of generations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    args = parser.parse_args()

    main(args)

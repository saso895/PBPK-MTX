# model.py  ---------------------------------------------------------------
"""
ODE system for adult MTX PBPK (Ogungbenro 2014).
Compartments order (length = 20):
[CP, CMU, CKID_V, CKID_T, CGLO, CPROX,
 CLIV_V, CLIV_T, CGUT_T,
 AST, ALUM, CENT,
 r_bile, CSK, CBM, CSP, CTH, CRE,
 CRBC_U, CRBC_B]
All concentrations in mg L-1, amounts where noted (AST, ALUM, r) in mg.
"""

import numpy as np
from scipy.integrate import solve_ivp
from data_mini import TISSUES, STOMACH_V, ENTEROCYTE_V, \
                 GLO_V, PT_V, GLO_Q, URINE_Q, GFR, BILE_TRANSIT_T, \
                 CL_PASS_LIV, CL_PASS_KID, FU, KA, CL_LI, CL_BILI, \
                 CL_SEC, PS_BC, Vmax_RBC, Km_RBC, KON_RBC, KOFF_RBC,\
                 QENT, VRBC

# index helpers
CP, CMU, CKV, CKT, CGLO, CPROX, \
CLVV, CLVT, CGUT, AST, ALUM, CENT, \
R_BILE, CSK, CBM, CSP, CTH, CRE, \
CRBCU, CRBCB = range(20)

# ──────────────────────────────── NEW BEGIN ─────────────────────────────
def _init_state(dose_mg, route="iv", kg=2.0, kt=0.25):
    """
    构造模型初值 y0 及参数字典 p
    与 simulate() 中原有逻辑一致，供 simulate_progress 等外部调用。
    """
    y0 = np.zeros(20)

    if route == "iv":
        y0[CP] = dose_mg / TISSUES["PLAS"].V
        #blood_V = TISSUES["PLAS"].V + 2.1       # 约等于 5 L 全血（70 kg 成人）
        #y0[CP] = dose_mg / blood_V
    elif route == "po":
        y0[AST] = dose_mg
    else:
        raise ValueError("route must be 'iv' or 'po'")

    p = dict(kg=kg, kt=kt)
    return y0, p
# ────────────────────────────────  NEW END  ─────────────────────────────

def pbpk_ode(t, y, dose_route, params):
    """Return dydt for the 20-state MTX PBPK system."""
    dy = np.zeros_like(y)

    # ---- plasma (Eq 1) ---------------------------------------------------
    vp = TISSUES["PLAS"].V
    q_tot = sum(tp.Q for tp in TISSUES.values() if tp.Q)     # systemic flows

    # helpers for tissue efflux terms: Q * (CT/Kp)
    def efflux(name, C):
        tp = TISSUES[name]
        return tp.Q * (C / tp.Kp)

    # Influx from perfused tissues
    influx = (efflux("LIV_V", y[CLVV]) +
              efflux("GUT_T", y[CGUT]) +
              TISSUES["SP"].Q * y[CSP]/TISSUES["SP"].Kp +
              efflux("KID_V", y[CKV]) )

    influx += (efflux("MU", y[CMU]) + efflux("SK", y[CSK]) +
               efflux("BM", y[CBM]) + efflux("TH", y[CTH]) +
               efflux("RE", y[CRE]))

    # Passive RBC efflux
    influx += PS_BC * (y[CRBCU]- y[CP])#/ VRBC

    # Elimination (hep + net sec) appears in CLVT & CKT, not plasma directly
    satur_uptake = Vmax_RBC * FU * y[CP] / (Km_RBC + FU * y[CP])

    dy[CP] = (influx -
              q_tot * y[CP] - VRBC*satur_uptake) / vp

    # ---- muscle (Eq 2)
    dy[CMU] = (TISSUES["MU"].Q *
               (y[CP] - y[CMU]/TISSUES["MU"].Kp)) / TISSUES["MU"].V

    # ---- kidney vascular & tissue (Eq 3)
    dy[CKV] = (TISSUES["KID_V"].Q*(y[CP]-y[CKV]) -
               CL_PASS_KID*FU*y[CKV] +
               CL_PASS_KID*FU*y[CKT]/TISSUES["KID_T"].Kp) / TISSUES["KID_V"].V

    dy[CKT] = (CL_PASS_KID*FU*(y[CKV]-y[CKT]/TISSUES["KID_T"].Kp) -
               CL_SEC*FU*y[CKT]/TISSUES["KID_T"].Kp) / TISSUES["KID_T"].V

    # glomerulus & proximal tubule
    dy[CGLO]  = (GFR*FU*y[CP] - GLO_Q*y[CGLO]) / GLO_V
    dy[CPROX] = (GLO_Q*y[CGLO] + CL_SEC*FU*y[CKT]/TISSUES["KID_T"].Kp -
                 URINE_Q*y[CPROX]) / PT_V

    # ---- liver vascular & tissue (Eq 4)
    Qgut = TISSUES["GUT_T"].Q  + QENT#TISSUES["PLAS"].Q*0  # ent + gut luminal portal
    dy[CLVV] = ((TISSUES["LIV_V"].Q*(y[CP]-y[CLVV])) -
                CL_PASS_LIV*FU*y[CLVV] +
                CL_PASS_LIV*FU*y[CLVT]/TISSUES["LIV_T"].Kp +
                Qgut*(y[CGUT]/TISSUES["GUT_T"].Kp - y[CLVV])) / TISSUES["LIV_V"].V

    dy[CLVT] = (CL_PASS_LIV*FU*(y[CLVV]-y[CLVT]/TISSUES["LIV_T"].Kp) -
                (CL_LI+CL_BILI)*FU*y[CLVT]/TISSUES["LIV_T"].Kp) / TISSUES["LIV_T"].V

    # ---- gut tissue (Eq 5 first part)
    dy[CGUT] = (TISSUES["GUT_T"].Q*(y[CP]-y[CGUT]/TISSUES["GUT_T"].Kp)) / TISSUES["GUT_T"].V

    # ---- stomach → lumen → enterocyte (Eq 5 second part + 6)
    dy[AST]  = -params["kg"] * y[AST]
    dy[ALUM] = params["kg"] * y[AST] - KA*y[ALUM] - params["kt"]*y[ALUM] + y[R_BILE]
    dy[CENT] = (KA*y[ALUM] - TISSUES["PLAS"].Q*0*y[CENT]) / ENTEROCYTE_V  # negligible flow term

    # ---- biliary transit (Eq 7)
    dy[R_BILE] = (CL_BILI*FU*y[CLVT]/TISSUES["LIV_T"].Kp - y[R_BILE]/BILE_TRANSIT_T)

    # ---- skin / BM / spleen / thymus / rest
    def tissue_1c(name, idx):
        tp = TISSUES[name]
        dy[idx] = tp.Q*(y[CP] - y[idx]/tp.Kp) / tp.V
    tissue_1c("SK", CSK)
    tissue_1c("BM", CBM)
    tissue_1c("SP", CSP)
    tissue_1c("TH", CTH)
    tissue_1c("RE", CRE)

    # ---- RBC (Eq 13)
    # dy[CRBCU] = (satur_uptake / TISSUES["PLAS"].V -
    #              PS_BC*y[CRBCU]/TISSUES["PLAS"].V -
    #              KON_RBC*y[CRBCU] + KOFF_RBC*y[CRBCB])
    dy[CRBCU] = (satur_uptake                       # mg · L-1 h-1
                 #- PS_BC*y[CRBCU]#/VRBC
                 -PS_BC*(y[CRBCU]-y[CP]) / VRBC 
                - KON_RBC*y[CRBCU] + KOFF_RBC*y[CRBCB])
    dy[CRBCB] = (KON_RBC*y[CRBCU] - KOFF_RBC*y[CRBCB])

    return dy

# ------------------------------------------------------------------------
def simulate(dose_mg, route="iv", t_end=48, n_points=400, kg=2.0, kt=0.25):
    """
    dose_mg : absolute dose (mg)
    route   : "iv" or "po"
    returns t, Cp (mg/L)
    """
    # initial state
    y0 = np.zeros(20)
    p  = dict(kg=kg, kt=kt)

    if route == "iv":
        y0[CP] = dose_mg / TISSUES["PLAS"].V
    elif route == "po":
        # assume received on empty stomach
        y0[AST] = dose_mg
    else:
        raise ValueError("route must be iv or po")

    sol = solve_ivp(pbpk_ode, [0, t_end], y0,
                    args=(route, p), dense_output=True, max_step=0.2)

    t = np.linspace(0, t_end, n_points)
    Cp = sol.sol(t)[CP]
    return t, Cp, sol

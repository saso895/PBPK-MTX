# checks.py  —— 质量平衡工具  (放在 PBPK-MTX 根目录)

import numpy as np
from pathlib import Path

def mass_balance(sol, dose_mg):
    """
    sol : odeint 返回的 y 矩阵，shape = (nt, 7)
          每一列都是“该室内药量 (mg)”或“累积排泄量 (mg)”
    dose_mg : 初始给药量 (mg)
    ----------
    返回   recovery ≈ 1 说明质量守恒
    """

    total_remaining = sol[-1, :].sum()      # 7 列全部加起来
    return total_remaining / dose_mg

def write_report(recovery, out_dir="saved_result"):
    Path(out_dir).mkdir(exist_ok=True)
    (Path(out_dir, "mass_balance.txt")
     .write_text(f"Mass balance recovery = {recovery:.4f}\n"))

#!/usr/bin/env python3
# ===============================================================
# compute_burnin_rhat.py
# ---------------------------------------------------------------
# 逐步计算累计 R‑hat(k) / ESS(k) ，并给出建议 burn‑in 步数
# 2025‑07‑24  by ChatGPT
# ===============================================================

import argparse, glob, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import xarray as xr

# ---------------------------- util -----------------------------

def load_chain(file_path):
    """
    兼容几种常见格式：
      • pandas DataFrame (to_numpy)
      • 直接存 numpy.ndarray
      • list / dict -> 转 numpy
    返回 ndarray shape = (draw , n_param)
    """
    try:
        obj = pd.read_pickle(file_path)
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy(), list(obj.columns)
        else:
            # 退一步：DataFrame 之外的对象
            arr = np.array(obj)
            return arr, None
    except Exception:
        # 直接 pickle.load
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy(), list(obj.columns)
        arr = np.array(obj)
        return arr, None


def build_idata(chains, param_names=None):
    """
    chains: list of ndarray, each shape = (draw , n_param)
    -> ArviZ InferenceData
    """
    arr = np.stack(chains, axis=0)  # (chain , draw , param)
    n_param = arr.shape[2]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_param)]
    else:
        # 容错：数量不对就强行生成
        if len(param_names) != n_param:
            param_names = [f"param_{i}" for i in range(n_param)]

    # xarray Dataset
    ds = xr.Dataset(
        {
            "theta": (("chain", "draw", "param"), arr)
        },
        coords={
            "chain": np.arange(arr.shape[0]),
            "draw":  np.arange(arr.shape[1]),
            "param": param_names
        }
    )
    return az.from_dict(posterior=ds)


# ------------------------- main logic --------------------------

def cumulative_diagnostics(idata, step=500, threshold=1.05, save_dir="diag_output"):
    """
    逐步计算 max R‑hat(k) / min ESS(k)
    返回 DataFrame
    """
    posterior = idata.posterior["theta"]  # dims: chain, draw, param
    n_draw = posterior.sizes["draw"]

    ks, max_rhats, min_ess = [], [], []
    suggested = None

    for k in range(step, n_draw + 1, step):
        subset = idata.sel(draw=slice(0, k - 1))
        rhats = az.rhat(subset, method="rank").to_array().values.flatten()
        ess_bulk = az.ess(subset, method="bulk").to_array().values.flatten()

        max_rhat = rhats.max()
        min_ess_k = ess_bulk.min()

        ks.append(k)
        max_rhats.append(max_rhat)
        min_ess.append(min_ess_k)

        if suggested is None and max_rhat < threshold:
            suggested = k

    df = pd.DataFrame(
        {"k": ks, "max_rhat": max_rhats, "min_ess": min_ess}
    )
    df.attrs["suggested_burnin"] = suggested
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(Path(save_dir) / "rhat_progress.csv", index=False)

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.plot(df["k"], df["max_rhat"], marker="o")
    plt.axhline(threshold, ls="--", lw=1, label=f"threshold={threshold}")
    if suggested:
        plt.axvline(suggested, color="red", ls=":", lw=1,
                    label=f"suggest burn‑in = {suggested}")
    plt.xlabel("Cumulative draws k")
    plt.ylabel("max R‑hat(k)")
    plt.title("R‑hat convergence diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "rhat_progress.png", dpi=300)
    plt.close()

    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute cumulative R‑hat(k) and suggest burn‑in"
    )
    p.add_argument("--chain_glob", required=True,
                   help="glob pattern for chain *.pkl, e.g. 'saved_result/chain*_draws*.pkl'")
    p.add_argument("--step", type=int, default=500,
                   help="step size (default 500)")
    p.add_argument("--threshold", type=float, default=1.05,
                   help="R‑hat threshold (default 1.05)")
    p.add_argument("--save_dir", default="diag_output",
                   help="directory to save CSV / PNG")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(args.chain_glob))
    if not files:
        sys.exit(f"[ERROR] No files matched: {args.chain_glob}")

    chains = []
    param_names_all = None
    for fp in files:
        arr, names = load_chain(fp)
        chains.append(arr)
        if names is not None:
            param_names_all = names  # 取第一条含列名的链

    # 对齐长度
    min_len = min(c.shape[0] for c in chains)
    chains = [c[:min_len] for c in chains]

    idata = build_idata(chains, param_names_all)
    df = cumulative_diagnostics(
        idata, step=args.step, threshold=args.threshold, save_dir=args.save_dir
    )

    sugg = df.attrs["suggested_burnin"]
    if sugg:
        print(f"✔  Suggested burn‑in = {sugg} draws "
              f"(first k with max R‑hat < {args.threshold})")
    else:
        print("✘  No k satisfied the threshold — chain likely not converged.")

    print(f"Results saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()

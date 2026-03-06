# plot_bo_convergence.py
# -*- coding: utf-8 -*-
"""
Plot Bayesian Optimization (or random search) convergence from bo_trace.csv.

CSV columns expected (written by your preprocess.py):
- iter, n_segments, compactness, mean_purity, mean_edge, mean_segments, mean_score

Usage examples:
  # 方式1：显式指定 CSV
  python plot_bo_convergence.py --csv "D:\L学术科研3\处理后图像\superpixels\_bo_log\bo_trace.csv" --grid

  # 方式2：不指定 --csv（将自动从 configs.superpixel_dir/_bo_log/bo_trace.csv 推断）
  python plot_bo_convergence.py --grid

  # 方式3：输出 PDF
  python plot_bo_convergence.py --csv "..." --out "...\bo_convergence.pdf"
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_WINDOWS_CSV = r"D:\L学术科研3\处理后图像\superpixels\_bo_log\bo_trace.csv"

def guess_csv_from_configs() -> Optional[str]:
    """Try to infer CSV path from configs.config.superpixel_dir."""
    try:
        from configs import config  # must be importable from CWD or PYTHONPATH
        return os.path.join(config.superpixel_dir, "_bo_log", "bo_trace.csv")
    except Exception:
        return None

def resolve_csv_path(user_csv: Optional[str]) -> str:
    """Resolve CSV path with the following priority:
       1) --csv argument
       2) configs.superpixel_dir/_bo_log/bo_trace.csv
       3) DEFAULT_WINDOWS_CSV (your provided path)
    """
    candidates = []
    if user_csv:
        candidates.append(user_csv)
    cfg_csv = guess_csv_from_configs()
    if cfg_csv:
        candidates.append(cfg_csv)
    candidates.append(DEFAULT_WINDOWS_CSV)

    for p in candidates:
        if p and os.path.exists(p):
            return p

    # If not found, raise with helpful message
    msg = [
        "[Error] bo_trace.csv not found.",
        "Tried the following locations:"
    ] + [f" - {p}" for p in candidates if p]
    msg += [
        "\nFixes:",
        "  • Ensure you've run preprocess.py with config.sp_log_trials=True",
        f"  • Or pass an explicit path: python {os.path.basename(__file__)} --csv \"<path-to>\\bo_trace.csv\"",
    ]
    raise FileNotFoundError("\n".join(msg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to bo_trace.csv")
    ap.add_argument("--out", default=None, help="Output figure path (.png/.pdf). Default: alongside CSV.")
    ap.add_argument("--title", default="Bayesian Optimization Convergence (higher is better)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--width", type=float, default=6.0)
    ap.add_argument("--height", type=float, default=4.0)
    ap.add_argument("--grid", action="store_true", help="Show grid")
    args = ap.parse_args()

    csv_path = resolve_csv_path(args.csv)
    out_path = args.out or os.path.join(os.path.dirname(csv_path), "bo_convergence.png")

    # Read CSV
    df = pd.read_csv(csv_path)
    # Normalize/validate columns
    if "iter" not in df.columns:
        if "iteration" in df.columns:
            df = df.rename(columns={"iteration": "iter"})
        else:
            raise ValueError(f"'iter' column not found in CSV. Columns: {list(df.columns)}")
    if "mean_score" not in df.columns:
        raise ValueError(f"'mean_score' column not found in CSV. Columns: {list(df.columns)}")

    df = df.sort_values("iter").reset_index(drop=True)
    # best-so-far in MAX sense (your mean_score is 'higher is better')
    df["best_so_far"] = df["mean_score"].cummax()

    # Plot
    plt.figure(figsize=(args.width, args.height))
    plt.plot(df["iter"], df["mean_score"], marker="o", linewidth=1.6, label="Score per iteration")
    plt.plot(df["iter"], df["best_so_far"], linestyle="--", linewidth=1.8, label="Best-so-far")

    # Annotate global best
    best_idx = df["best_so_far"].idxmax()
    best_iter = int(df.loc[best_idx, "iter"])
    best_score = float(df.loc[best_idx, "best_so_far"])
    plt.scatter([best_iter], [best_score], s=50, c="red", zorder=3)
    plt.text(best_iter, best_score, f"  best={best_score:.3f} @ iter {best_iter}",
             va="bottom", fontsize=9)

    # Axes & style (IEEE-like)
    plt.xlabel("Iteration", fontname="Times New Roman", fontsize=11)
    plt.ylabel("BO objective score (higher is better)", fontname="Times New Roman", fontsize=11)
    plt.title(args.title, fontsize=10)
    if args.grid:
        plt.grid(True, linestyle="--", alpha=0.35)

    ax = plt.gca()
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(fontsize=9)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"[OK] Saved convergence figure to: {out_path}")
    print(f"[INFO] Source CSV: {csv_path}")
    # Optional: print final best params if present
    maybe_cols = ["n_segments", "compactness"]
    if all(c in df.columns for c in maybe_cols):
        # Find row achieving best_so_far
        row = df.loc[best_idx, ["iter", "mean_score"] + maybe_cols]
        print("[INFO] Best trial row:")
        print(row.to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

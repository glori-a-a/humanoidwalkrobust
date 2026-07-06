"""Aggregate eval CSV: mean ± std over eval seeds; plot delay curves with error bars."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "success_rate",
    "mean_episode_length",
    "track_lin_vel_xy_exp",
    "mean_episode_return",
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["policy_source", "intervention", "intervention_value", "delay_steps", "delay_mode"]
    group_cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(group_cols, dropna=False)[METRICS].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def plot_fixed_delay(agg: pd.DataFrame, out_prefix: str, title: str):
    sub = agg[(agg["intervention"] == "delay") & (agg.get("delay_mode", "fixed") == "fixed")].copy()
    if sub.empty:
        sub = agg[agg["intervention"] == "delay"].copy()
    if sub.empty:
        print("No delay rows to plot.")
        return
    sub = sub.sort_values("delay_steps")
    x = sub["delay_steps"].values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, METRICS[:3]):
        mcol, scol = f"{metric}_mean", f"{metric}_std"
        if mcol not in sub.columns:
            continue
        y = sub[mcol].values
        yerr = sub[scol].values if scol in sub.columns else None
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, color="#2563eb")
        ax.set_xlabel("Action delay (sim steps)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if metric == "success_rate":
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = f"{out_prefix}_agg_fixed.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Eval protocol CSV (multi-seed rows).")
    parser.add_argument("--out", default=None, help="Output prefix (default: csv path without .csv).")
    parser.add_argument("--title", default="Delay robustness (mean ± std over eval seeds)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "delay_steps" not in df.columns and "intervention_value" in df.columns:
        df["delay_steps"] = df["intervention_value"].str.extract(r"^(\d+)").astype(float)

    agg = aggregate(df)
    out_prefix = args.out or args.csv.rsplit(".", 1)[0]
    agg_path = f"{out_prefix}_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")
    print(agg.to_string(index=False))
    plot_fixed_delay(agg, out_prefix, args.title)


if __name__ == "__main__":
    main()

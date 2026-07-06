"""Supervisor report figures — Part I (delay core) + Part II (velocity & training)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from parse_train_log import parse_log

MS_PER_STEP = 20.0
MAX_EP_STEPS = 2000

POLICIES = {
    "step0": {"label": "Step 0: author walk.pt", "color": "#6b7280", "marker": "^", "ls": "--"},
    "nominal": {"label": "Nominal (no delay DR)", "color": "#dc2626", "marker": "s", "ls": "-"},
    "strong": {"label": "Strong (delay DR U[0,5])", "color": "#2563eb", "marker": "o", "ls": "-"},
    "comp": {"label": "Delay-compensated", "color": "#059669", "marker": "D", "ls": "-"},
}

# TienKung walk_cfg defaults (training); eval locks lin_vel_x=1.0
TRAIN_VEL_RANGES = {
    "lin_vel_x (forward)": (-0.6, 1.0),
    "lin_vel_y (lateral)": (-0.5, 0.5),
    "ang_vel_z (yaw)": (-1.57, 1.57),
}
EVAL_VEL = {"lin_vel_x": 1.0, "lin_vel_y": 0.0, "ang_vel_z": 0.0}

TRAIN_BUDGET = {
    "num_envs": 2048,
    "max_iterations": 15000,
    "seed": 42,
    "nominal_delay_dr": "OFF",
    "strong_delay_dr": "U[0,5] steps (~0–100 ms)",
    "comp_delay_dr": "U[0,5] + predictor (Stage 1+2)",
    "actor_history": 10,
    "train_episode_s": 20,
    "eval_episode_s": 40,
    "eval_command": "lin_vel_x=1.0 m/s, lin_vel_y=0",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_agg(path: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["policy_key"] = key
    return df


def aggregate_raw_extra(path: Path) -> pd.DataFrame:
    """Add fall_rate and termination_penalty to agg from raw eval CSV."""
    df = pd.read_csv(path)
    df = df[df["intervention"] == "delay"].copy()
    df["fall_rate"] = df["fall_count"] / df["episodes"].clip(lower=1)
    group = ["delay_steps", "delay_mode"]
    agg = df.groupby(group).agg(
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        fall_rate_mean=("fall_rate", "mean"),
        fall_rate_std=("fall_rate", "std"),
        mean_episode_length_mean=("mean_episode_length", "mean"),
        mean_episode_length_std=("mean_episode_length", "std"),
        track_lin_vel_xy_exp_mean=("track_lin_vel_xy_exp", "mean"),
        track_lin_vel_xy_exp_std=("track_lin_vel_xy_exp", "std"),
        mean_episode_return_mean=("mean_episode_return", "mean"),
        mean_episode_return_std=("mean_episode_return", "std"),
        termination_penalty_mean=("termination_penalty", "mean"),
        termination_penalty_std=("termination_penalty", "std"),
    ).reset_index()
    return agg


def filter_fixed(df: pd.DataFrame, key: str) -> pd.DataFrame:
    sub = df[(df["delay_mode"] == "fixed")].copy()
    sub["policy_key"] = key
    return sub.sort_values("delay_steps")


def step0_rows(step0_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(step0_csv)
    df = df[df["intervention"] == "delay"].copy()
    df["delay_mode"] = "fixed"
    for col in [
        "success_rate",
        "mean_episode_length",
        "track_lin_vel_xy_exp",
        "mean_episode_return",
        "termination_penalty",
    ]:
        if col in df.columns:
            df[f"{col}_mean"] = df[col]
            df[f"{col}_std"] = 0.0
    df["fall_rate_mean"] = 1.0 - df["success_rate_mean"]
    df["fall_rate_std"] = 0.0
    df["policy_key"] = "step0"
    return df


def plot_metric(ax, dfs: list[pd.DataFrame], ycol: str, ylabel: str, ylim=None, legend=False):
    for df in dfs:
        key = df["policy_key"].iloc[0]
        style = POLICIES[key]
        x = df["delay_steps"].values
        y = df[f"{ycol}_mean"].values
        std_col = f"{ycol}_std"
        yerr = df[std_col].values if std_col in df.columns else None
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=f"{style['marker']}{style['ls']}",
            color=style["color"], label=style["label"],
            capsize=3, lw=1.8, ms=6,
        )
    ax.set_xlabel("Action delay (steps)")
    ax.set_xticks(range(0, 9))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(fontsize=8, loc="best", framealpha=0.92)


def add_ms_top(ax):
    sec = ax.secondary_xaxis("top", functions=(lambda s: s * MS_PER_STEP, lambda m: m / MS_PER_STEP))
    sec.set_xlabel("Delay (ms)")


# ── Part I: Delay core ──────────────────────────────────────────────

def part1_delay_core(
    nominal: pd.DataFrame, strong: pd.DataFrame, comp: pd.DataFrame, step0: pd.DataFrame, out: Path
):
    nf, sf, cf = (
        filter_fixed(nominal, "nominal"),
        filter_fixed(strong, "strong"),
        filter_fixed(comp, "comp"),
    )
    s0 = step0
    policy_dfs = [nf, sf, cf]

    # 1) Four-panel stability dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("success_rate", "Success rate (no fall, full 2000 steps)", (-0.05, 1.05)),
        ("fall_rate", "Fall rate (episodes ending before 2000 steps)", (-0.05, 1.05)),
        ("mean_episode_length", "Mean episode length (fall threshold = 2000)", None),
        ("termination_penalty", "Termination penalty (mean reward term)", None),
    ]
    for ax, (col, title, ylim) in zip(axes.ravel(), metrics):
        plot_metric(ax, policy_dfs, col, title, ylim=ylim, legend=(col == "success_rate"))
        if col == "mean_episode_length":
            ax.axhline(MAX_EP_STEPS, color="#16a34a", ls="--", lw=1.2, alpha=0.8)
            ax.text(0.5, MAX_EP_STEPS + 40, "success threshold (2000 steps)", fontsize=8, color="#16a34a")
    add_ms_top(axes[0, 0])
    fig.suptitle(
        "【核心】Part I — Delay 鲁棒性：平衡 / 跌倒 / 稳定性\n"
        "Fixed delay | Nominal vs Strong vs Comp | 3 seeds × 30 ep | eval: 1.0 m/s forward",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    p = out / "01_delay_stability_4panel.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 2) Step0 + nominal + strong + comp success (breakpoint)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_metric(ax, [s0, nf, sf, cf], "success_rate", "Success rate", ylim=(-0.05, 1.05), legend=True)
    ax.axvline(2, color="#dc2626", ls=":", alpha=0.5)
    ax.annotate("Nominal / Step0\nbreakpoint ≈ d=2 (40ms)", xy=(2, 0.05), xytext=(3.2, 0.35),
                fontsize=8, color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626"))
    add_ms_top(ax)
    ax.set_title("Delay breakpoint — author vs self-trained policies")
    fig.tight_layout()
    p = out / "02_delay_breakpoint_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 3) Episode length with fall zone shading
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(0, MAX_EP_STEPS, alpha=0.08, color="red", label="Fall / unstable zone")
    ax.axhline(MAX_EP_STEPS, color="#16a34a", ls="--", lw=1.5, label="Full episode (stable)")
    plot_metric(ax, policy_dfs, "mean_episode_length", "Mean episode length (steps)", legend=True)
    add_ms_top(ax)
    ax.set_title("跌倒阈值：episode length < 2000 ⇒ 判定摔倒 (success=0)")
    fig.tight_layout()
    p = out / "03_fall_threshold_episode_length.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 4) Fixed vs uniform — stability under time-varying delay
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mode, title in zip(axes, ["fixed", "uniform"], ["Fixed delay d", "Uniform delay U[0,d]"]):
        nsub = nominal[nominal["delay_mode"] == mode]
        ssub = strong[strong["delay_mode"] == mode]
        csub = comp[comp["delay_mode"] == mode]
        if mode == "uniform":
            nsub = nsub[nsub["delay_steps"].isin([2, 4, 6, 8])]
            ssub = ssub[ssub["delay_steps"].isin([2, 4, 6, 8])]
            csub = csub[csub["delay_steps"].isin([2, 4, 6, 8])]
        nsub = nsub.copy(); ssub = ssub.copy(); csub = csub.copy()
        nsub["policy_key"] = "nominal"; ssub["policy_key"] = "strong"; csub["policy_key"] = "comp"
        plot_metric(ax, [nsub, ssub, csub], "success_rate", "Success rate", ylim=(-0.05, 1.05), legend=True)
        ax.set_title(title)
    fig.suptitle("【核心】Fixed vs time-varying delay — 运动稳定性", fontsize=12)
    fig.tight_layout()
    p = out / "04_delay_fixed_vs_uniform.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 5) Combined return + tracking under delay (stability quality)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_metric(axes[0], policy_dfs, "mean_episode_return", "Mean episode return", legend=True)
    plot_metric(axes[1], policy_dfs, "track_lin_vel_xy_exp", "Velocity tracking reward", legend=True)
    add_ms_top(axes[0])
    fig.suptitle("【核心】Delay 下综合运动质量 (return & velocity tracking)", fontsize=12)
    fig.tight_layout()
    p = out / "05_delay_motion_quality.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ── Part II: Velocity & training ────────────────────────────────────

def part2_velocity_and_training(
    nominal: pd.DataFrame, strong: pd.DataFrame, comp: pd.DataFrame, step0: pd.DataFrame,
    train_logs: list[Path], out: Path,
):
    nf = filter_fixed(nominal, "nominal")
    sf = filter_fixed(strong, "strong")
    cf = filter_fixed(comp, "comp")
    policy_dfs = [nf, sf, cf]

    # 1) Training velocity command ranges (config reference)
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(TRAIN_VEL_RANGES.keys())
    lows = [TRAIN_VEL_RANGES[n][0] for n in names]
    highs = [TRAIN_VEL_RANGES[n][1] for n in names]
    y = np.arange(len(names))
    ax.barh(y, [h - l for l, h in zip(lows, highs)], left=lows, height=0.5, color="#93c5fd", edgecolor="#2563eb")
    ax.scatter([EVAL_VEL.get("lin_vel_x", 0), EVAL_VEL.get("lin_vel_y", 0), EVAL_VEL.get("ang_vel_z", 0)],
               y, color="#dc2626", s=120, zorder=5, marker="*", label="Eval locked command")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("m/s  (ang_vel_z: rad/s)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.legend(loc="lower right")
    ax.set_title(
        "【补充】训练期速度命令采样范围 (walk_cfg 默认)\n"
        "训练：多方向随机；评估：固定 lin_vel_x=1.0 m/s 直行"
    )
    fig.tight_layout()
    p = out / "01_velocity_command_ranges.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 2) Velocity tracking vs delay (eval at 1 m/s command)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_metric(ax, policy_dfs, "track_lin_vel_xy_exp",
                "Velocity tracking reward (eval @ 1.0 m/s forward)", legend=True)
    add_ms_top(ax)
    ax.set_title("【补充】各 delay 下速度跟踪状态 (命令固定 1.0 m/s)")
    fig.tight_layout()
    p = out / "02_velocity_tracking_vs_delay.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 3) d=0 trade-off: nominal vs strong vs comp vs step0
    d0_rows = []
    for name, df in [("Step 0", step0[step0["delay_steps"] == 0]),
                     ("Nominal", nf[nf["delay_steps"] == 0]),
                     ("Strong", sf[sf["delay_steps"] == 0]),
                     ("Comp", cf[cf["delay_steps"] == 0])]:
        r = df.iloc[0]
        d0_rows.append({
            "policy": name,
            "success": r["success_rate_mean"] * 100,
            "tracking": r["track_lin_vel_xy_exp_mean"],
            "return": r["mean_episode_return_mean"],
        })
    d0 = pd.DataFrame(d0_rows)
    x = np.arange(len(d0))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, d0["success"], w, label="Success %", color="#16a34a")
    ax.bar(x, d0["tracking"] * 100, w, label="Tracking reward ×100", color="#2563eb")
    ax.bar(x + w, d0["return"], w, label="Episode return", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(d0["policy"])
    ax.set_title("【补充】无延迟 (d=0) 各策略速度跟踪 & 综合表现")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "03_d0_velocity_tradeoff.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 4) Training config table figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    rows = [[k, str(v)] for k, v in TRAIN_BUDGET.items()]
    table = ax.table(cellText=rows, colLabels=["Parameter", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title(
        "【补充】训练配置快照 (nominal / strong 共享预算)\n"
        "注：训练曲线需从 RunPod Volume 拉回 *train*.log 后自动绘制",
        fontsize=11, pad=20,
    )
    fig.tight_layout()
    p = out / "04_training_config_snapshot.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # 5) Training curves if logs present
    if train_logs:
        fig, axes = plt.subplots(len(train_logs), 2, figsize=(12, 4 * len(train_logs)), squeeze=False)
        for i, log in enumerate(train_logs):
            df = parse_log(log)
            if df.empty:
                continue
            label = log.stem
            axes[i, 0].plot(df["iteration"], df.get("mean_episode_length", pd.Series(dtype=float)), color="#2563eb")
            axes[i, 0].axhline(1000, color="#16a34a", ls="--", alpha=0.7, label="train ep_len target ~1000")
            axes[i, 0].set_title(f"{label} — Mean episode length")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
            if "mean_reward" in df.columns:
                axes[i, 1].plot(df["iteration"], df["mean_reward"], color="#dc2626")
                axes[i, 1].set_title(f"{label} — Mean reward")
                axes[i, 1].set_xlabel("Iteration")
                axes[i, 1].grid(True, alpha=0.3)
        fig.suptitle("【补充】全维度训练指标 — 收敛曲线", fontsize=12, y=1.01)
        fig.tight_layout()
        p = out / "05_training_convergence_curves.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")
    else:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "训练收敛曲线待补：请将 RunPod 上的\n"
            "  baseline_train_baseline_seed42_v3.log\n"
            "  strong_baseline_train_strong_delay_rand_seed42_v3.log\n"
            "放入 results/training_logs/ 后重跑本脚本。\n\n"
            "解析命令: python scripts/parse_train_log.py <log文件>",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#fef3c7", edgecolor="#d97706"),
        )
        p = out / "05_training_convergence_PENDING.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p} (training logs not local)")


def export_tables(nominal, strong, comp, step0, out: Path):
    rows = []
    for tag, df in [("step0", step0), ("nominal", nominal), ("strong", strong), ("comp", comp)]:
        sub = df[df["delay_mode"] == "fixed"] if "delay_mode" in df.columns else df
        for _, r in sub.iterrows():
            rows.append({
                "section": "PartI_delay_core",
                "policy": tag,
                "delay_steps": int(r["delay_steps"]),
                "delay_ms": int(r["delay_steps"] * MS_PER_STEP),
                "delay_mode": r.get("delay_mode", "fixed"),
                "success_pct": round(100 * r["success_rate_mean"], 1),
                "fall_rate_pct": round(100 * r.get("fall_rate_mean", 1 - r["success_rate_mean"]), 1),
                "mean_episode_length": round(r["mean_episode_length_mean"], 1),
                "velocity_tracking": round(r["track_lin_vel_xy_exp_mean"], 4),
                "episode_return": round(r["mean_episode_return_mean"], 2),
                "termination_penalty": round(r.get("termination_penalty_mean", float("nan")), 4),
            })
    pd.DataFrame(rows).to_csv(out / "full_metrics_table.csv", index=False)
    print(f"Saved: {out / 'full_metrics_table.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/csv")
    parser.add_argument("--out", default="results/figures/delay_report")
    args = parser.parse_args()

    root = repo_root()
    results = root / args.results
    base_out = root / args.out
    part1 = base_out / "part1_delay_core"
    part2 = base_out / "part2_supplementary"
    part1.mkdir(parents=True, exist_ok=True)
    part2.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 150,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    nominal = aggregate_raw_extra(results / "eval_nominal.csv")
    strong = aggregate_raw_extra(results / "eval_strong.csv")
    comp = aggregate_raw_extra(results / "eval_compensated.csv")
    step0 = step0_rows(results / "step0_author_walk.csv")

    train_log_dir = results / "training_logs"
    train_logs = sorted(train_log_dir.glob("*.log")) if train_log_dir.exists() else []
    if not train_logs:
        train_logs = sorted(results.glob("*train*.log"))

    part1_delay_core(nominal, strong, comp, step0, part1)
    part2_velocity_and_training(nominal, strong, comp, step0, train_logs, part2)
    export_tables(nominal, strong, comp, step0, base_out)

    print(f"\nFigures: {base_out}")


if __name__ == "__main__":
    main()

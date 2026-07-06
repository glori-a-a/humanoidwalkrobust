"""Thesis main figure: nominal vs strong vs delay-compensated (fixed delay, mean ± std)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MS_PER_STEP = 20.0

POLICIES = {
    "nominal": {
        "label": "Nominal (no delay DR)",
        "color": "#dc2626",
        "marker": "s",
    },
    "strong": {
        "label": "Strong (delay DR U[0,5])",
        "color": "#2563eb",
        "marker": "o",
    },
    "comp": {
        "label": "Delay-compensated",
        "color": "#059669",
        "marker": "D",
    },
    "history20": {
        "label": "History-20 ablation",
        "color": "#ea580c",
        "marker": "^",
    },
}


def load_fixed_delay(agg_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(agg_csv)
    sub = df[(df["intervention"] == "delay") & (df["delay_mode"] == "fixed")].copy()
    if sub.empty:
        raise ValueError(f"No fixed-delay rows in {agg_csv}")
    return sub.sort_values("delay_steps")


def plot_success_rate(ax, series: dict[str, pd.DataFrame]) -> None:
    for key, sub in series.items():
        style = POLICIES[key]
        x = sub["delay_steps"].values
        y = sub["success_rate_mean"].values
        yerr = sub["success_rate_std"].values
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{style['marker']}-",
            color=style["color"],
            label=style["label"],
            capsize=4,
            linewidth=2,
            markersize=7,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Action delay (sim steps)")
    ax.set_ylabel("Success rate")
    ax.set_title("Stability under fixed action delay")
    ax.set_xticks(range(0, 9))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda s: s * MS_PER_STEP, lambda ms: ms / MS_PER_STEP),
    )
    secax.set_xlabel("Approx. delay (ms)")


def plot_metric(ax, series: dict[str, pd.DataFrame], metric: str, title: str, ylabel: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    for key, sub in series.items():
        style = POLICIES[key]
        x = sub["delay_steps"].values
        y = sub[mean_col].values
        yerr = sub[std_col].values
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{style['marker']}-",
            color=style["color"],
            label=style["label"],
            capsize=3,
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xlabel("Action delay (sim steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 9))
    ax.grid(True, alpha=0.3)


def annotate_breakpoints(ax, series: dict[str, pd.DataFrame]) -> None:
    for key, sub in series.items():
        cliff = sub[sub["success_rate_mean"] < 0.5]
        if cliff.empty:
            continue
        d = int(cliff.iloc[0]["delay_steps"])
        color = POLICIES[key]["color"]
        ax.axvline(d, color=color, linestyle=":", alpha=0.45, linewidth=1.2)
        y_text = {"nominal": 0.12, "strong": 0.78, "history20": 0.58, "comp": 0.42}.get(key, 0.45)
        ax.annotate(
            f"{POLICIES[key]['label'].split('(')[0].strip()}\nbreakpoint ≈ d={d}",
            xy=(d, 0.5),
            xytext=(d + 0.35, y_text),
            fontsize=8,
            color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nominal",
        default="results/csv/eval_nominal_agg.csv",
        help="Aggregated CSV for nominal policy.",
    )
    parser.add_argument(
        "--strong",
        default="results/csv/eval_strong_agg.csv",
        help="Aggregated CSV for strong policy.",
    )
    parser.add_argument(
        "--comp",
        default="results/csv/eval_compensated_agg.csv",
        help="Aggregated CSV for delay-compensated policy.",
    )
    parser.add_argument(
        "--history20",
        default="results/csv/eval_history20_agg.csv",
        help="Aggregated CSV for history-20 ablation.",
    )
    parser.add_argument(
        "--out",
        default="results/figures/thesis_main_delay_curve",
        help="Output path prefix (without extension).",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    nominal_path = (repo / args.nominal).resolve()
    strong_path = (repo / args.strong).resolve()
    comp_path = (repo / args.comp).resolve()
    history_path = (repo / args.history20).resolve()
    out_prefix = (repo / args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    series = {
        "nominal": load_fixed_delay(nominal_path),
        "strong": load_fixed_delay(strong_path),
        "history20": load_fixed_delay(history_path),
        "comp": load_fixed_delay(comp_path),
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    # Single-panel main figure (paper primary)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plot_success_rate(ax, series)
    annotate_breakpoints(ax, series)
    fig.suptitle(
        "Delay robustness: nominal vs DR vs history-20 vs delay-compensated\n"
        "(frozen policies, 3 eval seeds, 30 episodes/point)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    main_png = f"{out_prefix}.png"
    main_pdf = f"{out_prefix}.pdf"
    fig.savefig(main_png, dpi=200, bbox_inches="tight")
    fig.savefig(main_pdf, bbox_inches="tight")
    print(f"Saved: {main_png}")
    print(f"Saved: {main_pdf}")
    plt.close(fig)

    # Three-panel supplementary-style figure
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    plot_success_rate(axes[0], series)
    plot_metric(
        axes[1],
        series,
        "mean_episode_length",
        "Episode length",
        "Mean episode length (steps)",
    )
    plot_metric(
        axes[2],
        series,
        "track_lin_vel_xy_exp",
        "Velocity tracking",
        "track_lin_vel_xy_exp",
    )
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.suptitle(
        "Four-way comparison — fixed delay sweep (d = 0…8 steps, 20 ms/step)",
        fontsize=11,
    )
    fig.tight_layout()
    sup_png = f"{out_prefix}_3panel.png"
    fig.savefig(sup_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {sup_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()

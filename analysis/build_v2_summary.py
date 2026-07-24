"""Build v2 thesis tables + delay curves from local agg CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "csv"
FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# label -> path
SOURCES = {
    "Strong-15k": CSV / "eval_strong_agg.csv",
    "Strong-30k": CSV / "eval_strong_30k_agg.csv",
    "v2a U[0,5]": CSV / "eval_queue_ablation_agg.csv",
    "v2a U[0,8]+2k": CSV / "eval_u08_model_17900_agg.csv",
    "v2a U[0,8]+5k": CSV / "eval_u08_model_20898_agg.csv",
    "History-20": CSV / "eval_history20_agg.csv",
    "Comp v1": CSV / "eval_compensated_agg.csv",
}

STYLES = {
    "Strong-15k": {"color": "#94a3b8", "marker": "o", "ls": "--"},
    "Strong-30k": {"color": "#2563eb", "marker": "s", "ls": "-"},
    "v2a U[0,5]": {"color": "#f59e0b", "marker": "^", "ls": ":"},
    "v2a U[0,8]+2k": {"color": "#059669", "marker": "D", "ls": "-"},
    "v2a U[0,8]+5k": {"color": "#10b981", "marker": "v", "ls": "-."},
    "History-20": {"color": "#ea580c", "marker": "x", "ls": "--"},
    "Comp v1": {"color": "#7c3aed", "marker": "P", "ls": "--"},
}


def load_fixed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    sub = df[(df["intervention"] == "delay") & (df["delay_mode"] == "fixed")].copy()
    return sub.sort_values("delay_steps")


def pct(x) -> str:
    return f"{100 * float(x):.1f}%"


def build_main_table(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    delays = [0, 2, 4, 5, 6, 8]
    rows = []
    for name, df in series.items():
        row = {"method": name}
        for d in delays:
            hit = df[df["delay_steps"] == d]
            if hit.empty:
                row[f"d{d}"] = ""
            else:
                m = float(hit["success_rate_mean"].iloc[0])
                s = float(hit["success_rate_std"].iloc[0])
                row[f"d{d}"] = f"{pct(m)}" if s < 1e-9 else f"{pct(m)}±{100*s:.1f}"
        # tracking at d=4 if present
        hit4 = df[df["delay_steps"] == 4]
        if not hit4.empty and "track_lin_vel_xy_exp_mean" in hit4.columns:
            row["track_d4"] = f"{float(hit4['track_lin_vel_xy_exp_mean'].iloc[0]):.3f}"
        else:
            row["track_d4"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_uniform_table(sources: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in sources.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sub = df[(df["intervention"] == "delay") & (df["delay_mode"] == "uniform")]
        row = {"method": name}
        for d in (5, 8):
            hit = sub[sub["delay_steps"] == d]
            if hit.empty:
                row[f"U[0,{d}]"] = ""
            else:
                m = float(hit["success_rate_mean"].iloc[0])
                s = float(hit["success_rate_std"].iloc[0])
                row[f"U[0,{d}]"] = f"{pct(m)}" if s < 1e-9 else f"{pct(m)}±{100*s:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(series: dict[str, pd.DataFrame], keys: list[str], out: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    metrics = [
        ("success_rate", "Success rate", (-0.05, 1.05)),
        ("mean_episode_length", "Episode length (steps)", (0, 2100)),
        ("track_lin_vel_xy_exp", "Velocity tracking reward", None),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for key in keys:
            if key not in series:
                continue
            df = series[key]
            st = STYLES[key]
            mean_c, std_c = f"{metric}_mean", f"{metric}_std"
            if mean_c not in df.columns:
                continue
            ax.errorbar(
                df["delay_steps"],
                df[mean_c],
                yerr=df[std_c] if std_c in df.columns else None,
                fmt=f"{st['marker']}{st['ls']}",
                color=st["color"],
                label=key,
                capsize=3,
                lw=2,
                ms=6,
            )
        ax.set_xlabel("Action delay (steps)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, 9))
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')}")


def build_clamp_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    g = (
        df.groupby("clamp_mode", as_index=False)
        .agg(
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            length_mean=("mean_episode_length", "mean"),
            track_mean=("track_lin_vel_xy_exp", "mean"),
        )
        .sort_values("success_mean", ascending=False)
    )
    g["success"] = g.apply(
        lambda r: f"{pct(r.success_mean)}"
        if pd.isna(r.success_std) or r.success_std < 1e-9
        else f"{pct(r.success_mean)}±{100 * r.success_std:.1f}",
        axis=1,
    )
    return g[["clamp_mode", "success", "length_mean", "track_mean"]]


def main():
    series = {}
    for name, path in SOURCES.items():
        if path.exists():
            series[name] = load_fixed(path)
        else:
            print(f"skip missing {path}")

    main_tbl = build_main_table(series)
    main_path = CSV / "thesis_v2_success_table.csv"
    main_tbl.to_csv(main_path, index=False)
    print(f"Saved {main_path}")
    print(main_tbl.to_string(index=False))

    uni = build_uniform_table(SOURCES)
    uni_path = CSV / "thesis_v2_uniform_table.csv"
    uni.to_csv(uni_path, index=False)
    print(f"Saved {uni_path}")

    clamp_path = CSV / "eval_queue_clamp.csv"
    if clamp_path.exists():
        clamp = build_clamp_table(clamp_path)
        out = CSV / "thesis_v2_clamp_table.csv"
        clamp.to_csv(out, index=False)
        print(f"Saved {out}")
        print(clamp.to_string(index=False))

    # core fair comparison figure
    plot_curves(
        series,
        ["Strong-15k", "Strong-30k", "v2a U[0,5]", "v2a U[0,8]+2k"],
        FIG / "v2_main_delay_curve",
        "v2 fair comparison — fixed delay (3 eval seeds)",
    )
    # include legacy methods
    plot_curves(
        series,
        ["Strong-30k", "History-20", "Comp v1", "v2a U[0,8]+2k"],
        FIG / "v2_vs_legacy_delay_curve",
        "v2a U[0,8]+2k vs legacy methods — fixed delay",
    )
    # FT progress
    plot_curves(
        series,
        ["v2a U[0,5]", "v2a U[0,8]+2k", "v2a U[0,8]+5k"],
        FIG / "v2a_u08_finetune_progress",
        "v2a U[0,8] fine-tune progress (+0 / +2k / +5k)",
    )


if __name__ == "__main__":
    main()

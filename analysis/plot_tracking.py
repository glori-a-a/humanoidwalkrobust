"""Plot vx / vy / yaw rate vs time (commanded vs actual)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

POLICY_STYLE = {
    "nominal": {"color": "#dc2626", "label": "Nominal"},
    "strong": {"color": "#2563eb", "label": "Strong (delay DR)"},
    "history20": {"color": "#ea580c", "label": "History-20 ablation"},
    "comp": {"color": "#059669", "label": "Delay-compensated"},
}

DELAY_ORDER = [0, 2, 4]

AXES = [
    ("vx", "vx_cmd", "vx_actual", "vx (m/s)", (-0.2, 1.3)),
    ("vy", "vy_cmd", "vy_actual", "vy (m/s)", (-0.4, 0.4)),
    ("yaw", "yaw_cmd", "yaw_actual", "yaw rate (rad/s)", (-1.0, 1.0)),
]


def load_series(data_dir: Path) -> pd.DataFrame:
    # use tracking_* only; vx_* are legacy duplicates of the same episodes
    tracking = sorted(data_dir.glob("tracking_*.csv"))
    if tracking:
        return pd.concat([pd.read_csv(f) for f in tracking], ignore_index=True)
    legacy = sorted(data_dir.glob("vx_*.csv"))
    if not legacy:
        raise FileNotFoundError(f"No tracking_*.csv in {data_dir}")
    return pd.concat([pd.read_csv(f) for f in legacy], ignore_index=True)


def plot_overlay_axis(df: pd.DataFrame, axis_key: str, cmd_col: str, act_col: str, ylabel: str, ylim, out: Path, title: str):
    if cmd_col not in df.columns or act_col not in df.columns:
        print(f"Skip {out.name}: missing {cmd_col}/{act_col}")
        return

    delays = [d for d in DELAY_ORDER if d in df["delay_steps"].unique()]
    fig, axes = plt.subplots(1, len(delays), figsize=(5 * len(delays), 4), squeeze=False)

    for j, delay in enumerate(delays):
        ax = axes[0][j]
        sub_d = df[df["delay_steps"] == delay]
        ref = sub_d[sub_d["policy"] == sub_d["policy"].iloc[0]]
        ax.plot(ref["time_s"], ref[cmd_col], "k--", lw=1.2, alpha=0.7, label="command")

        for policy, style in POLICY_STYLE.items():
            sub = sub_d[sub_d["policy"] == policy]
            if sub.empty:
                continue
            ax.plot(sub["time_s"], sub[act_col], color=style["color"], lw=1.8, label=style["label"])
            if sub["fell"].iloc[0]:
                ax.axvline(sub["time_s"].iloc[-1], color=style["color"], ls=":", lw=1, alpha=0.6)

        ax.set_title(f"d={delay} ({delay * 20} ms)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


def plot_combined(df: pd.DataFrame, out: Path):
    delays = [d for d in DELAY_ORDER if d in df["delay_steps"].unique()]
    fig, axes = plt.subplots(len(AXES), len(delays), figsize=(5 * len(delays), 3.5 * len(AXES)), squeeze=False)

    for i, (key, cmd_col, act_col, ylabel, ylim) in enumerate(AXES):
        if cmd_col not in df.columns:
            continue
        for j, delay in enumerate(delays):
            ax = axes[i][j]
            sub_d = df[df["delay_steps"] == delay]
            ref = sub_d[sub_d["policy"] == sub_d["policy"].iloc[0]]
            ax.plot(ref["time_s"], ref[cmd_col], "k--", lw=1.0, alpha=0.6)
            for policy, style in POLICY_STYLE.items():
                sub = sub_d[sub_d["policy"] == policy]
                if sub.empty:
                    continue
                ax.plot(sub["time_s"], sub[act_col], color=style["color"], lw=1.4, label=style["label"])
            if i == 0:
                ax.set_title(f"d={delay} ({delay * 20} ms)")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(AXES) - 1:
                ax.set_xlabel("time (s)")
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            if i == 0 and j == len(delays) - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Velocity tracking time-series: vx, vy, yaw (lin_vel_x=1.0, lin_vel_y=0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    df = load_series(args.data_dir)
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overlay_axis(
        df, "vx", "vx_cmd", "vx_actual", "vx (m/s)", (-0.2, 1.3),
        out_dir / "vx_tracking_overlay.png", "Forward velocity vx (supervisor request)",
    )
    plot_overlay_axis(
        df, "vy", "vy_cmd", "vy_actual", "vy (m/s)", (-0.4, 0.4),
        out_dir / "vy_tracking_overlay.png", "Lateral velocity vy (command = 0)",
    )
    plot_overlay_axis(
        df, "yaw", "yaw_cmd", "yaw_actual", "yaw rate (rad/s)", (-1.0, 1.0),
        out_dir / "yaw_tracking_overlay.png", "Yaw rate (command = 0)",
    )
    plot_combined(df, out_dir / "vel_tracking_all.png")


if __name__ == "__main__":
    main()

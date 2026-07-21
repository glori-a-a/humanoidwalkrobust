"""Plot v2 tracking: Strong-30k vs v2a U[0,8]+2k across command directions."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

POLICY_STYLE = {
    "strong30k": {"color": "#2563eb", "label": "Strong-30k"},
    "v2a_u08": {"color": "#059669", "label": "v2a U[0,8]+2k"},
}

DIR_SPECS = [
    ("forward", 1.0, 0.0, "Forward (vx=1.0, vy=0)"),
    ("lateral", 0.0, 0.4, "Lateral (vx=0, vy=0.4)"),
    ("diagonal", 0.8, 0.3, "Diagonal (vx=0.8, vy=0.3)"),
]

AXES = [
    ("vx_cmd", "vx_actual", "vx (m/s)"),
    ("vy_cmd", "vy_actual", "vy (m/s)"),
    ("yaw_cmd", "yaw_actual", "yaw rate (rad/s)"),
]


def load_tracking(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("tracking_*.csv"))
    if not files:
        raise FileNotFoundError(f"No tracking_*.csv in {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # parse filename fallbacks
        m = re.search(r"tracking_(.+)_d(\d+)_vx([-\d.]+)_vy([-\d.]+)", f.name)
        if m and "policy" not in df.columns:
            df["policy"] = m.group(1)
        if m and "delay_steps" not in df.columns:
            df["delay_steps"] = int(m.group(2))
        if m and "lin_vel_x" not in df.columns:
            df["lin_vel_x"] = float(m.group(3))
            df["lin_vel_y"] = float(m.group(4))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_direction(df: pd.DataFrame, name: str, vx: float, vy: float, title: str, out: Path):
    sub = df[(df["lin_vel_x"].round(3) == round(vx, 3)) & (df["lin_vel_y"].round(3) == round(vy, 3))]
    if sub.empty:
        print(f"skip empty {name}")
        return
    delays = sorted(sub["delay_steps"].unique())
    fig, axes = plt.subplots(len(AXES), len(delays), figsize=(4.2 * len(delays), 3.2 * len(AXES)), squeeze=False)
    for i, (cmd, act, ylabel) in enumerate(AXES):
        for j, d in enumerate(delays):
            ax = axes[i][j]
            sd = sub[sub["delay_steps"] == d]
            ref = sd.iloc[0:1]
            # command from first series
            first = sd[sd["policy"] == sd["policy"].iloc[0]]
            ax.plot(first["time_s"], first[cmd], "k--", lw=1.1, alpha=0.7, label="command")
            for pol, style in POLICY_STYLE.items():
                p = sd[sd["policy"] == pol]
                if p.empty:
                    continue
                ax.plot(p["time_s"], p[act], color=style["color"], lw=1.5, label=style["label"])
                if "fell" in p.columns and int(p["fell"].iloc[0]) == 1:
                    ax.axvline(p["time_s"].iloc[-1], color=style["color"], ls=":", lw=1, alpha=0.6)
            if i == 0:
                ax.set_title(f"d={d} ({d * 20} ms)")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(AXES) - 1:
                ax.set_xlabel("time (s)")
            ax.grid(True, alpha=0.25)
            if i == 0 and j == len(delays) - 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')}")


def plot_rmse_summary(df: pd.DataFrame, out: Path):
    rows = []
    for _, g in df.groupby(["policy", "delay_steps", "lin_vel_x", "lin_vel_y"], sort=False):
        rows.append(
            {
                "policy": g["policy"].iloc[0],
                "delay_steps": int(g["delay_steps"].iloc[0]),
                "lin_vel_x": float(g["lin_vel_x"].iloc[0]),
                "lin_vel_y": float(g["lin_vel_y"].iloc[0]),
                "fell": int(g["fell"].iloc[0]) if "fell" in g.columns else 0,
                "rmse_vx": float(((g["vx_actual"] - g["vx_cmd"]) ** 2).mean() ** 0.5),
                "rmse_vy": float(((g["vy_actual"] - g["vy_cmd"]) ** 2).mean() ** 0.5),
                "episode_steps": int(g["episode_steps"].iloc[0]) if "episode_steps" in g.columns else len(g),
            }
        )
    summary = pd.DataFrame(rows)
    csv_path = out.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # bar chart: RMSE vx+vy at d=0 and d=4 for each direction
    dirs = [(1.0, 0.0, "fwd"), (0.0, 0.4, "lat"), (0.8, 0.3, "diag")]
    delays = [0, 4]
    fig, axes = plt.subplots(1, len(delays), figsize=(10, 3.6), sharey=True)
    for ax, d in zip(axes, delays):
        cats, s30, v2a = [], [], []
        for vx, vy, lab in dirs:
            cats.append(lab)
            for pol, bucket in (("strong30k", s30), ("v2a_u08", v2a)):
                hit = summary[
                    (summary["policy"] == pol)
                    & (summary["delay_steps"] == d)
                    & (summary["lin_vel_x"].round(3) == round(vx, 3))
                    & (summary["lin_vel_y"].round(3) == round(vy, 3))
                ]
                if hit.empty:
                    bucket.append(0.0)
                else:
                    bucket.append(float(hit["rmse_vx"].iloc[0] + hit["rmse_vy"].iloc[0]))
        x = range(len(cats))
        ax.bar([i - 0.18 for i in x], s30, width=0.36, color="#2563eb", label="Strong-30k")
        ax.bar([i + 0.18 for i in x], v2a, width=0.36, color="#059669", label="v2a U[0,8]+2k")
        ax.set_xticks(list(x))
        ax.set_xticklabels(cats)
        ax.set_title(f"d={d}")
        ax.set_ylabel("RMSE vx+vy (m/s)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Velocity tracking RMSE by command direction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_name(out.stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.png')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.out_dir or (Path("results/figures") / "tracking_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tracking(args.data_dir)
    for name, vx, vy, title in DIR_SPECS:
        plot_direction(df, name, vx, vy, title, out_dir / f"tracking_{name}")
    plot_rmse_summary(df, out_dir / "tracking_rmse_by_direction")


if __name__ == "__main__":
    main()

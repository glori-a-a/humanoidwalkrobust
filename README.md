# Humanoid Learning Control

**MSc dissertation — redesigning the walking controller for action-delay robustness**

Real humanoid stacks suffer **action delay** (network, inference, actuation). A nominal AMP-PPO walker fails once delay exceeds ~40 ms. This repo implements a **controller redesign** — not just delay domain randomisation (DR), but a **two-stage learning pipeline** that predicts delayed observations and fine-tunes a compensated walking policy.

<p align="center">
  <img src="assets/demo_delay_walk.gif" alt="TienKung humanoid walking with the delay-compensated controller under action delay" width="640"/>
  <br/>
  <em>TienKung humanoid walking in simulation. Swap in an Isaac Lab clip of the delay-compensated policy at fixed delay (e.g. 4 steps) once recorded — see below.</em>
</p>

| Baseline | What it does |
|----------|----------------|
| **Nominal** | Standard walk policy, no delay DR |
| **Strong (DR)** | Trained with uniform delay DR U[0, 5] steps |
| **Delay-compensated (this work)** | Stage 1 observation predictor + Stage 2 fine-tuned policy under the same DR budget |

<p align="center">
  <img src="results/figures/thesis_main_delay_curve.png" alt="Success rate vs action delay for nominal, DR, and delay-compensated policies" width="720"/>
</p>

The redesigned controller (**green**) keeps high success rate at delays where the nominal policy collapses and outperforms delay DR alone at large delays.

## Stack

- Isaac Sim 4.5, Isaac Lab 2.1
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) (clone separately; not included here)
- Python 3.10, CUDA GPU for training and eval

## Install into TienKung-Lab

Copy files from this repo into your TienKung-Lab tree (same relative paths):

```
rsl_rl/modules/predictor.py
rsl_rl/modules/delay_policy.py          -> also register as ActorCriticDelayPredictor in modules/__init__.py
legged_lab/envs/tienkung/walk_compensated_cfg.py
legged_lab/envs/tienkung/walk_history_ablation_cfg.py
legged_lab/envs/tienkung/env_patch.py     -> run once to patch tienkung_env.py
legged_lab/scripts/*.py
```

Register tasks in `legged_lab/envs/__init__.py` (see comments in the cfg files). Register `delay_policy.ActorCriticDelayPredictor` in `rsl_rl/rsl_rl/modules/__init__.py` and runners.

## Training (on GPU server)

From `TienKung-Lab/` with conda env active:

```bash
# Strong baseline (delay DR U[0,5])
python legged_lab/scripts/train_strong_baseline.py --task=walk --headless --num_envs=2048 --max_iterations=15000 --seed=42

# Stage 1 — observation predictor (supervised)
python legged_lab/scripts/train_predictor.py --task=walk --headless --num_envs=2048 --seed=42 --predictor_path=...

# Stage 2 — compensated policy (load predictor + fine-tune)  *** controller redesign ***
python legged_lab/scripts/train_compensated_policy.py --task=walk_delay_comp --headless --num_envs=2048 --max_iterations=15000 --predictor_path=... --init_policy_path=...

# Ablation: long history (h=20), same DR, no predictor
python legged_lab/scripts/train_long_history.py --task=walk_history_ablation --headless --num_envs=2048 --max_iterations=15000 --seed=42
```

Checkpoints stay under `TienKung-Lab/logs/` (not tracked in git).

## Evaluation

```bash
python legged_lab/scripts/eval_robustness.py --headless --task walk \
  --policy_path /path/to/policy.pt --intervention delay --action_delay_steps 4 --delay_mode fixed

python legged_lab/scripts/aggregate_eval.py results/csv/eval_nominal.csv --out results/csv/eval_nominal_agg
```

## Figures (local)

```bash
python analysis/plot_main_results.py
python analysis/plot_tracking.py results/csv/tracking --out_dir results/figures
python analysis/plot_delay_report.py
```

Replace the README demo GIF after recording Isaac clips (e.g. `comp_d4` vs `strong_d4`):

```bash
python analysis/mp4_to_gif.py path/to/comp_d4.mp4 assets/demo_delay_walk.gif --fps 12 --width 640
```

## Results in this repo

| File | Description |
|------|-------------|
| `results/csv/eval_nominal*.csv` | Nominal baseline eval |
| `results/csv/eval_strong*.csv` | Delay DR baseline eval |
| `results/csv/eval_compensated*.csv` | Predictor + fine-tuned policy eval |
| `results/csv/eval_history20*.csv` | Long-history ablation (when available) |
| `results/figures/` | Main paper figures |

Training budget: 2048 envs, 15k iterations, seed 42 for all trained policies.

## Author

[glori-a-a](https://github.com/glori-a-a)

## Citation

If you use this code, cite the TienKung-Lab paper and your dissertation reference as appropriate.

# Failure and debugging record

This log records failures that are supported by code or results already in the
repository. It must not be extended with hypothetical hardware incidents.

## F-001 — Nominal controller loses robustness under action delay

- Environment: Isaac Lab simulation
- Symptom: the nominal policy degrades sharply once action delay increases;
  the repository's main evaluation describes failure beyond approximately
  40 ms.
- Evidence: `results/csv/eval_nominal.csv`,
  `results/csv/eval_nominal_agg.csv` and the main delay figures.
- Engineering response: introduced explicit delay-domain randomisation,
  observation history baselines and queue-aware policy inputs.
- Scope: simulator evidence only.

## F-002 — Action-queue off-by-one risk

- Environment: pure tensor/unit-test model of the action delay buffer.
- Risk: including the newly issued action in the pre-execution queue, reversing
  queue order, or selecting `u_(t-d+1)` instead of the currently applied
  `u_(t-d)` would make the policy observation inconsistent with execution.
- Evidence: `rsl_rl/modules/delay_queue.py` and
  `tests/test_delay_queue_order.py`.
- Engineering response: documented a canonical decision-time extraction rule
  and added fixed/mixed-delay tests.
- Scope: logic verification; not a hardware timing measurement.

## F-003 — Training-range generalisation at larger delays

- Environment: Isaac Lab simulation.
- Symptom: policies trained under a narrower delay distribution do not
  automatically retain the same performance at larger fixed delays.
- Evidence: `results/csv/eval_queue_clamp.csv`,
  `eval_u08_model_*_agg.csv` and `results/figures/v2a_u08_finetune_progress.*`.
- Engineering response: evaluated U[0,8] fine-tuning and queue-clamp ablations
  instead of reporting a single favourable operating point.
- Scope: simulator evidence only.

## F-004 — Predictor-feedback v2b path is incomplete

- Environment: source review.
- Symptom: `PredictorFeedbackActorCritic.actor_input()` explicitly raises
  `NotImplementedError` when `use_predictor=True`.
- Evidence: `rsl_rl/modules/predictor_feedback_actor_critic.py`.
- Required next step: connect the residual-dynamics rollout and future-state
  encoder, add tests, and rerun the fair-delay evaluation.
- Scope: known limitation; the queue-aware v2a path must not be described as a
  completed predictor-feedback controller.

## Hardware failures

None recorded. No physical TienKung run has been completed from this repository.


#!/usr/bin/env python3
"""Patch the pinned official TienKung StateMLP for this repository's policy.

The official controller already builds the same 75-value observation frame and
keeps ten frames (750 floats). The delay predictor is part of the exported
PyTorch/OpenVINO graph, so C++ feeds the raw delayed history exactly once; it
does not run a second predictor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_UPSTREAM = "1f2b8b8071398d65bc2b085570b2520449b633e9"
TARGET = Path("x_humanoid_rl_sdk/src/robot_FSM/FSMStateImpl.cpp")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    source = args.deploy_root / TARGET
    if not source.exists():
        raise FileNotFoundError(f"Official source not found: {source}")

    text = source.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "const int obs_num = 750;",
        """// Contract shared with ActorCriticDelayPredictor:
// 75 values/frame x 10 delayed frames -> embedded predictor -> 20 actions.
constexpr int kObsFrameSize = 75;
constexpr int kObsHistoryLength = 10;
constexpr int obs_num = kObsFrameSize * kObsHistoryLength;
constexpr int kPolicyActionSize = 20;""",
    )

    # The exported network has an explicit batch dimension. The original
    # upstream tensor was one-dimensional ({750}), which is incompatible with
    # an OpenVINO model whose input is [1, 750].
    text = replace_once(
        text,
        "ov::Tensor ov_in_tensor0(ov::element::f32, ov::Shape{obs_num}, input_vec.data());",
        "ov::Tensor ov_in_tensor0(ov::element::f32, ov::Shape{1, obs_num}, input_vec.data());",
    )

    text = replace_once(
        text,
        '  model = core.read_model(mlp_path + ".xml",  mlp_path + ".bin");\n  compiled_model = core.compile_model(model, "CPU");',
        '''  model = core.read_model(mlp_path + ".xml", mlp_path + ".bin");

  if (model->inputs().size() != 1 || model->outputs().size() != 1) {
    throw std::runtime_error("Delay policy must have exactly one input and one output");
  }
  const auto input_shape = model->input(0).get_partial_shape();
  const auto output_shape = model->output(0).get_partial_shape();
  if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic() ||
      input_shape.rank().get_length() != 2 || output_shape.rank().get_length() != 2 ||
      input_shape[0].is_dynamic() || input_shape[1].is_dynamic() ||
      output_shape[0].is_dynamic() || output_shape[1].is_dynamic() ||
      input_shape[0].get_length() != 1 || input_shape[1].get_length() != obs_num ||
      output_shape[0].get_length() != 1 ||
      output_shape[1].get_length() != kPolicyActionSize) {
    throw std::runtime_error(
        "Delay-compensated policy contract mismatch: expected [1,750] -> [1,20]");
  }

  compiled_model = core.compile_model(model, "CPU");''',
    )

    text = replace_once(
        text,
        "          std::vector<float> new_obs(input_vec.begin(), input_vec.begin() + 75);\n"
        "          std::move(input_vec.begin() + 75, input_vec.end(), input_vec.begin());\n"
        "          std::copy(new_obs.begin(), new_obs.end(), input_vec.end() - 75);",
        "          std::vector<float> new_obs(input_vec.begin(), input_vec.begin() + kObsFrameSize);\n"
        "          std::move(input_vec.begin() + kObsFrameSize, input_vec.end(), input_vec.begin());\n"
        "          std::copy(new_obs.begin(), new_obs.end(), input_vec.end() - kObsFrameSize);",
    )

    text = replace_once(
        text,
        "    ov::Tensor ov_out_tensor = infer_request.get_output_tensor();\n"
        "    const float *ov_out_data = ov_out_tensor.data<float>();",
        "    ov::Tensor ov_out_tensor = infer_request.get_output_tensor();\n"
        "    if (ov_out_tensor.get_size() != kPolicyActionSize) {\n"
        "      throw std::runtime_error(\"Policy output changed at runtime; expected 20 actions\");\n"
        "    }\n"
        "    const float *ov_out_data = ov_out_tensor.data<float>();",
    )

    if args.check:
        print(f"Patch applies cleanly to {source}")
        return

    source.write_text(text, encoding="utf-8")
    print(f"Patched {source}")
    print(f"Pinned upstream: {EXPECTED_UPSTREAM}")


if __name__ == "__main__":
    main()

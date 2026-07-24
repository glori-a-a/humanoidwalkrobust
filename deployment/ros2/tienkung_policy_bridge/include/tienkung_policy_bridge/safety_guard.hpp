#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tienkung_policy_bridge {

enum class Mode { kStartup, kStandby, kActive, kFault };

struct JointLimit {
  double lower{};
  double upper{};
  double default_position{};
};

struct GuardParameters {
  double action_scale{0.25};
  double max_target_velocity{1.0};
  double state_timeout_s{0.1};
};

struct GuardResult {
  Mode mode{Mode::kStartup};
  std::vector<double> target;
  std::string reason;
};

class SafetyGuard {
 public:
  SafetyGuard(std::vector<JointLimit> limits, GuardParameters parameters)
      : limits_(std::move(limits)),
        parameters_(parameters),
        last_target_(limits_.size(), 0.0) {
    if (limits_.empty()) {
      throw std::invalid_argument("joint limits must not be empty");
    }
    if (!(parameters_.action_scale > 0.0) ||
        !(parameters_.max_target_velocity > 0.0) ||
        !(parameters_.state_timeout_s > 0.0)) {
      throw std::invalid_argument("guard parameters must be positive");
    }
    for (std::size_t i = 0; i < limits_.size(); ++i) {
      const auto& limit = limits_[i];
      if (!Finite(limit.lower) || !Finite(limit.upper) ||
          !Finite(limit.default_position) || !(limit.lower < limit.upper) ||
          limit.default_position < limit.lower ||
          limit.default_position > limit.upper) {
        throw std::invalid_argument("invalid joint limit");
      }
      last_target_[i] = limit.default_position;
    }
  }

  void UpdateState(double stamp_s, const std::vector<double>& positions) {
    if (!Finite(stamp_s) || positions.size() != limits_.size() ||
        !AllFinite(positions)) {
      Fault("invalid joint state");
      return;
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (positions[i] < limits_[i].lower ||
          positions[i] > limits_[i].upper) {
        Fault("measured joint outside configured limits");
        return;
      }
    }
    state_stamp_s_ = stamp_s;
    latest_position_ = positions;
    state_available_ = true;
    if (mode_ == Mode::kStartup) {
      mode_ = Mode::kStandby;
    }
  }

  bool Enable(double now_s) {
    if (mode_ == Mode::kFault || !state_available_ || !Finite(now_s) ||
        now_s - state_stamp_s_ > parameters_.state_timeout_s ||
        now_s < state_stamp_s_) {
      return false;
    }
    last_target_ = latest_position_;
    mode_ = Mode::kActive;
    last_command_stamp_s_ = now_s;
    command_time_available_ = true;
    return true;
  }

  void Standby() {
    if (mode_ != Mode::kFault) {
      mode_ = Mode::kStandby;
    }
  }

  GuardResult Command(const std::vector<double>& normalized_action,
                      double now_s) {
    if (mode_ != Mode::kActive) {
      return {mode_, last_target_, "guard is not active"};
    }
    if (!state_available_ || !Finite(now_s) ||
        now_s < state_stamp_s_ ||
        now_s - state_stamp_s_ > parameters_.state_timeout_s) {
      Fault("stale or invalid robot state");
      return {mode_, last_target_, fault_reason_};
    }
    if (normalized_action.size() != limits_.size() ||
        !AllFinite(normalized_action)) {
      Fault("invalid policy action");
      return {mode_, last_target_, fault_reason_};
    }
    if (!command_time_available_ || now_s <= last_command_stamp_s_) {
      Fault("non-positive command timestep");
      return {mode_, last_target_, fault_reason_};
    }

    const double dt = now_s - last_command_stamp_s_;
    const double max_delta = parameters_.max_target_velocity * dt;
    std::vector<double> target(limits_.size(), 0.0);
    for (std::size_t i = 0; i < limits_.size(); ++i) {
      const double action = Clamp(normalized_action[i], -1.0, 1.0);
      const double desired = Clamp(
          limits_[i].default_position + parameters_.action_scale * action,
          limits_[i].lower, limits_[i].upper);
      target[i] = Clamp(desired, last_target_[i] - max_delta,
                        last_target_[i] + max_delta);
      target[i] = Clamp(target[i], limits_[i].lower, limits_[i].upper);
    }
    last_target_ = target;
    last_command_stamp_s_ = now_s;
    return {Mode::kActive, target, "ok"};
  }

  Mode mode() const { return mode_; }
  const std::string& fault_reason() const { return fault_reason_; }

 private:
  static bool Finite(double value) { return std::isfinite(value); }

  static bool AllFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return Finite(value); });
  }

  static double Clamp(double value, double lower, double upper) {
    return std::min(std::max(value, lower), upper);
  }

  void Fault(std::string reason) {
    mode_ = Mode::kFault;
    fault_reason_ = std::move(reason);
  }

  std::vector<JointLimit> limits_;
  GuardParameters parameters_;
  std::vector<double> last_target_;
  std::vector<double> latest_position_;
  Mode mode_{Mode::kStartup};
  std::string fault_reason_;
  bool state_available_{false};
  bool command_time_available_{false};
  double state_stamp_s_{0.0};
  double last_command_stamp_s_{0.0};
};

}  // namespace tienkung_policy_bridge

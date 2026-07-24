#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tienkung_policy_bridge {

enum class Mode { startup, standby, active, fault };

struct JointLimit {
  double lower;
  double upper;
  double default_position;
};

struct GuardSettings {
  double action_scale;
  double max_target_velocity;
  double state_timeout;
};

struct GuardState {
  std::vector<JointLimit> limits;
  GuardSettings settings;
  Mode mode{Mode::startup};
  std::string fault;
  std::vector<double> position;
  std::vector<double> last_target;
  double state_time{0.0};
  double last_command_time{0.0};
  bool has_state{false};
  bool has_command_time{false};
};

struct GuardResult {
  Mode mode;
  std::vector<double> target;
  std::string reason;
};

inline bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

inline double clip(double value, double lower, double upper) {
  return std::min(std::max(value, lower), upper);
}

inline void set_fault(GuardState& guard, const std::string& reason) {
  guard.mode = Mode::fault;
  guard.fault = reason;
}

inline GuardState make_guard(const std::vector<JointLimit>& limits,
                             const GuardSettings& settings) {
  if (limits.empty()) {
    throw std::invalid_argument("joint limits must not be empty");
  }
  if (settings.action_scale <= 0.0 ||
      settings.max_target_velocity <= 0.0 ||
      settings.state_timeout <= 0.0) {
    throw std::invalid_argument("guard settings must be positive");
  }

  GuardState guard;
  guard.limits = limits;
  guard.settings = settings;
  for (const auto& limit : limits) {
    if (!std::isfinite(limit.lower) || !std::isfinite(limit.upper) ||
        !std::isfinite(limit.default_position) ||
        limit.lower >= limit.upper ||
        limit.default_position < limit.lower ||
        limit.default_position > limit.upper) {
      throw std::invalid_argument("invalid joint limit");
    }
    guard.last_target.push_back(limit.default_position);
  }
  return guard;
}

inline bool update_state(GuardState& guard, double stamp,
                         const std::vector<double>& position) {
  if (!std::isfinite(stamp) ||
      position.size() != guard.limits.size() ||
      !all_finite(position)) {
    set_fault(guard, "invalid joint state");
    return false;
  }

  for (std::size_t i = 0; i < position.size(); ++i) {
    if (position[i] < guard.limits[i].lower ||
        position[i] > guard.limits[i].upper) {
      set_fault(guard, "joint position outside limits");
      return false;
    }
  }

  guard.position = position;
  guard.state_time = stamp;
  guard.has_state = true;
  if (guard.mode == Mode::startup) {
    guard.mode = Mode::standby;
  }
  return true;
}

inline bool enable(GuardState& guard, double now) {
  if (guard.mode == Mode::fault || !guard.has_state ||
      !std::isfinite(now)) {
    return false;
  }
  double age = now - guard.state_time;
  if (age < 0.0 || age > guard.settings.state_timeout) {
    return false;
  }

  guard.last_target = guard.position;
  guard.last_command_time = now;
  guard.has_command_time = true;
  guard.mode = Mode::active;
  return true;
}

inline void standby(GuardState& guard) {
  if (guard.mode != Mode::fault) {
    guard.mode = Mode::standby;
  }
}

inline GuardResult make_command(GuardState& guard,
                                const std::vector<double>& action,
                                double now) {
  if (guard.mode != Mode::active) {
    return {guard.mode, guard.last_target, "guard is not active"};
  }

  double age = now - guard.state_time;
  if (!std::isfinite(now) || age < 0.0 ||
      age > guard.settings.state_timeout) {
    set_fault(guard, "robot state is stale");
    return {guard.mode, guard.last_target, guard.fault};
  }

  if (action.size() != guard.limits.size() || !all_finite(action)) {
    set_fault(guard, "invalid policy action");
    return {guard.mode, guard.last_target, guard.fault};
  }

  double dt = now - guard.last_command_time;
  if (!guard.has_command_time || dt <= 0.0) {
    set_fault(guard, "invalid command time");
    return {guard.mode, guard.last_target, guard.fault};
  }

  double max_change = guard.settings.max_target_velocity * dt;
  std::vector<double> target;
  target.reserve(action.size());
  for (std::size_t i = 0; i < action.size(); ++i) {
    double value = clip(action[i], -1.0, 1.0);
    double desired = guard.limits[i].default_position +
                     guard.settings.action_scale * value;
    desired = clip(desired, guard.limits[i].lower, guard.limits[i].upper);
    desired = clip(desired,
                   guard.last_target[i] - max_change,
                   guard.last_target[i] + max_change);
    target.push_back(
        clip(desired, guard.limits[i].lower, guard.limits[i].upper));
  }

  guard.last_target = target;
  guard.last_command_time = now;
  return {Mode::active, target, "ok"};
}

}

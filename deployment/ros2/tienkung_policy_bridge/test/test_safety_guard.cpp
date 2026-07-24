#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "tienkung_policy_bridge/safety_guard.hpp"

using tienkung_policy_bridge::GuardParameters;
using tienkung_policy_bridge::JointLimit;
using tienkung_policy_bridge::Mode;
using tienkung_policy_bridge::SafetyGuard;

int main() {
  const std::vector<JointLimit> limits{{-1.0, 1.0, 0.0},
                                       {-0.5, 0.5, 0.1}};
  const GuardParameters parameters{0.5, 1.0, 0.1};

  SafetyGuard guard(limits, parameters);
  assert(!guard.Enable(1.0));
  guard.UpdateState(1.0, {0.0, 0.1});
  assert(guard.mode() == Mode::kStandby);
  assert(guard.Enable(1.0));

  const auto limited = guard.Command({4.0, -4.0}, 1.02);
  assert(limited.mode == Mode::kActive);
  assert(std::abs(limited.target.at(0) - 0.02) < 1e-12);
  assert(std::abs(limited.target.at(1) - 0.08) < 1e-12);

  const auto stale = guard.Command({0.0, 0.0}, 1.2);
  assert(stale.mode == Mode::kFault);

  SafetyGuard invalid_action_guard(limits, parameters);
  invalid_action_guard.UpdateState(2.0, {0.0, 0.1});
  assert(invalid_action_guard.Enable(2.0));
  const auto invalid_action = invalid_action_guard.Command(
      {std::numeric_limits<double>::quiet_NaN(), 0.0}, 2.01);
  assert(invalid_action.mode == Mode::kFault);

  SafetyGuard measured_start_guard(limits, parameters);
  measured_start_guard.UpdateState(3.0, {0.4, -0.2});
  assert(measured_start_guard.Enable(3.0));
  const auto measured_start =
      measured_start_guard.Command({0.0, 0.0}, 3.01);
  assert(std::abs(measured_start.target.at(0) - 0.39) < 1e-12);
  assert(std::abs(measured_start.target.at(1) - (-0.19)) < 1e-12);

  std::cout << "C++ safety guard tests passed\n";
  return 0;
}

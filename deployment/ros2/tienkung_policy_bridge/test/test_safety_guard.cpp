#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "tienkung_policy_bridge/safety_guard.hpp"

using namespace tienkung_policy_bridge;

int main() {
  std::vector<JointLimit> limits{
      {-1.0, 1.0, 0.0},
      {-0.5, 0.5, 0.1},
  };
  GuardSettings settings{0.5, 1.0, 0.1};

  auto guard = make_guard(limits, settings);
  assert(!enable(guard, 1.0));
  assert(update_state(guard, 1.0, {0.0, 0.1}));
  assert(guard.mode == Mode::standby);
  assert(enable(guard, 1.0));

  auto limited = make_command(guard, {4.0, -4.0}, 1.02);
  assert(limited.mode == Mode::active);
  assert(std::abs(limited.target[0] - 0.02) < 1e-12);
  assert(std::abs(limited.target[1] - 0.08) < 1e-12);

  auto stale = make_command(guard, {0.0, 0.0}, 1.2);
  assert(stale.mode == Mode::fault);

  auto bad_action_guard = make_guard(limits, settings);
  update_state(bad_action_guard, 2.0, {0.0, 0.1});
  assert(enable(bad_action_guard, 2.0));
  auto bad_action = make_command(
      bad_action_guard,
      {std::numeric_limits<double>::quiet_NaN(), 0.0},
      2.01);
  assert(bad_action.mode == Mode::fault);

  auto measured_guard = make_guard(limits, settings);
  update_state(measured_guard, 3.0, {0.4, -0.2});
  assert(enable(measured_guard, 3.0));
  auto measured = make_command(measured_guard, {0.0, 0.0}, 3.01);
  assert(std::abs(measured.target[0] - 0.39) < 1e-12);
  assert(std::abs(measured.target[1] + 0.19) < 1e-12);

  std::cout << "safety guard tests passed\n";
  return 0;
}

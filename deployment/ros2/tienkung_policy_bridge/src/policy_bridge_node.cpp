#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "tienkung_policy_bridge/safety_guard.hpp"

namespace tienkung_policy_bridge {

class PolicyBridgeNode final : public rclcpp::Node {
 public:
  PolicyBridgeNode() : Node("tienkung_policy_bridge") {
    const auto joint_names =
        declare_parameter<std::vector<std::string>>("joint_names", {});
    const auto lower = declare_parameter<std::vector<double>>("joint_lower", {});
    const auto upper = declare_parameter<std::vector<double>>("joint_upper", {});
    const auto defaults =
        declare_parameter<std::vector<double>>("joint_default", {});
    mapping_verified_ =
        declare_parameter<bool>("hardware_mapping_verified", false);

    if (joint_names.empty() || joint_names.size() != lower.size() ||
        lower.size() != upper.size() || upper.size() != defaults.size()) {
      throw std::runtime_error(
          "joint_names/lower/upper/default must be non-empty and equal length");
    }
    if (std::any_of(joint_names.begin(), joint_names.end(),
                    [](const std::string& name) {
                      return name.rfind("UNVERIFIED_", 0) == 0;
                    })) {
      mapping_verified_ = false;
    }
    joint_names_ = joint_names;

    std::vector<JointLimit> limits;
    limits.reserve(joint_names.size());
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
      limits.push_back({lower[i], upper[i], defaults[i]});
    }
    GuardParameters parameters;
    parameters.action_scale =
        declare_parameter<double>("action_scale", 0.25);
    parameters.max_target_velocity =
        declare_parameter<double>("max_target_velocity", 1.0);
    parameters.state_timeout_s =
        declare_parameter<double>("state_timeout_s", 0.1);
    guard_ = std::make_unique<SafetyGuard>(limits, parameters);

    command_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "/policy/safe_joint_targets", rclcpp::QoS(1));
    status_pub_ =
        create_publisher<std_msgs::msg::String>("/policy/guard_status", 10);
    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
          OnJointState(*message);
        });
    action_sub_ =
        create_subscription<std_msgs::msg::Float64MultiArray>(
            "/policy/action_normalized", rclcpp::QoS(1),
            [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
              latest_action_ = message->data;
              action_available_ = true;
            });
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/policy/enable", rclcpp::QoS(1),
        [this](std_msgs::msg::Bool::ConstSharedPtr message) {
          if (!message->data) {
            guard_->Standby();
            PublishStatus("standby requested");
            return;
          }
          if (!mapping_verified_) {
            PublishStatus(
                "enable rejected: hardware joint mapping is not verified");
            return;
          }
          if (!guard_->Enable(NowSeconds())) {
            PublishStatus("enable rejected by safety guard");
          }
        });

    const auto rate_hz = declare_parameter<double>("control_rate_hz", 50.0);
    if (!(rate_hz > 0.0)) {
      throw std::runtime_error("control_rate_hz must be positive");
    }
    timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate_hz),
        [this]() { Tick(); });

    if (!mapping_verified_) {
      RCLCPP_WARN(
          get_logger(),
          "Hardware mapping is UNVERIFIED; activation is intentionally locked");
    }
  }

 private:
  double NowSeconds() const { return get_clock()->now().seconds(); }

  void OnJointState(const sensor_msgs::msg::JointState& message) {
    if (message.name.size() != message.position.size()) {
      PublishStatus("invalid JointState: name/position size mismatch");
      return;
    }
    std::unordered_map<std::string, double> position_by_name;
    for (std::size_t i = 0; i < message.name.size(); ++i) {
      position_by_name[message.name[i]] = message.position[i];
    }
    std::vector<double> ordered;
    ordered.reserve(joint_names_.size());
    for (const auto& name : joint_names_) {
      const auto it = position_by_name.find(name);
      if (it == position_by_name.end()) {
        PublishStatus("JointState missing expected joint: " + name);
        return;
      }
      ordered.push_back(it->second);
    }
    const double stamp =
        rclcpp::Time(message.header.stamp).nanoseconds() == 0
            ? NowSeconds()
            : rclcpp::Time(message.header.stamp).seconds();
    guard_->UpdateState(stamp, ordered);
  }

  void Tick() {
    if (!action_available_) {
      return;
    }
    const auto result = guard_->Command(latest_action_, NowSeconds());
    if (result.mode != Mode::kActive) {
      PublishStatus(result.reason);
      return;
    }
    std_msgs::msg::Float64MultiArray command;
    command.data = result.target;
    command_pub_->publish(command);
  }

  void PublishStatus(const std::string& text) {
    std_msgs::msg::String status;
    status.data = text;
    status_pub_->publish(status);
  }

  bool mapping_verified_{false};
  bool action_available_{false};
  std::vector<std::string> joint_names_;
  std::vector<double> latest_action_;
  std::unique_ptr<SafetyGuard> guard_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr action_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace tienkung_policy_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<tienkung_policy_bridge::PolicyBridgeNode>());
  rclcpp::shutdown();
  return 0;
}


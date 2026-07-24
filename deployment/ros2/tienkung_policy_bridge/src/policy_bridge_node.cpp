#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "tienkung_policy_bridge/safety_guard.hpp"

namespace tienkung_policy_bridge {

class PolicyBridgeNode : public rclcpp::Node {
 public:
  PolicyBridgeNode() : Node("tienkung_policy_bridge") {
    joint_names_ =
        declare_parameter<std::vector<std::string>>("joint_names", {});
    auto lower =
        declare_parameter<std::vector<double>>("joint_lower", {});
    auto upper =
        declare_parameter<std::vector<double>>("joint_upper", {});
    auto defaults =
        declare_parameter<std::vector<double>>("joint_default", {});
    mapping_verified_ =
        declare_parameter<bool>("hardware_mapping_verified", false);

    if (joint_names_.empty() ||
        joint_names_.size() != lower.size() ||
        lower.size() != upper.size() ||
        upper.size() != defaults.size()) {
      throw std::runtime_error("invalid joint configuration");
    }

    for (const auto& name : joint_names_) {
      if (name.rfind("UNVERIFIED_", 0) == 0) {
        mapping_verified_ = false;
      }
    }

    std::vector<JointLimit> limits;
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      limits.push_back({lower[i], upper[i], defaults[i]});
    }

    GuardSettings settings{
        declare_parameter<double>("action_scale", 0.25),
        declare_parameter<double>("max_target_velocity", 1.0),
        declare_parameter<double>("state_timeout_s", 0.1),
    };
    guard_ = make_guard(limits, settings);

    command_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "/policy/safe_joint_targets", 1);
    status_pub_ =
        create_publisher<std_msgs::msg::String>("/policy/guard_status", 10);

    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states",
        rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
          read_joint_state(*message);
        });

    action_sub_ =
        create_subscription<std_msgs::msg::Float64MultiArray>(
            "/policy/action_normalized",
            1,
            [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
              latest_action_ = message->data;
              has_action_ = true;
            });

    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/policy/enable",
        1,
        [this](std_msgs::msg::Bool::ConstSharedPtr message) {
          set_enabled(message->data);
        });

    double rate = declare_parameter<double>("control_rate_hz", 50.0);
    if (rate <= 0.0) {
      throw std::runtime_error("control rate must be positive");
    }

    timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate),
        [this]() { update(); });
  }

 private:
  double now_seconds() {
    return get_clock()->now().seconds();
  }

  void publish_status(const std::string& text) {
    std_msgs::msg::String message;
    message.data = text;
    status_pub_->publish(message);
  }

  void read_joint_state(const sensor_msgs::msg::JointState& message) {
    if (message.name.size() != message.position.size()) {
      publish_status("invalid joint state");
      return;
    }

    std::unordered_map<std::string, double> values;
    for (std::size_t i = 0; i < message.name.size(); ++i) {
      values[message.name[i]] = message.position[i];
    }

    std::vector<double> position;
    for (const auto& name : joint_names_) {
      auto item = values.find(name);
      if (item == values.end()) {
        publish_status("missing joint: " + name);
        return;
      }
      position.push_back(item->second);
    }

    double stamp = rclcpp::Time(message.header.stamp).seconds();
    if (rclcpp::Time(message.header.stamp).nanoseconds() == 0) {
      stamp = now_seconds();
    }
    update_state(guard_, stamp, position);
  }

  void set_enabled(bool value) {
    if (!value) {
      standby(guard_);
      publish_status("standby");
      return;
    }
    if (!mapping_verified_) {
      publish_status("joint mapping is not verified");
      return;
    }
    if (!enable(guard_, now_seconds())) {
      publish_status("enable rejected");
    }
  }

  void update() {
    if (!has_action_) {
      return;
    }

    auto result = make_command(guard_, latest_action_, now_seconds());
    if (result.mode != Mode::active) {
      publish_status(result.reason);
      return;
    }

    std_msgs::msg::Float64MultiArray message;
    message.data = result.target;
    command_pub_->publish(message);
  }

  bool mapping_verified_{false};
  bool has_action_{false};
  GuardState guard_;
  std::vector<std::string> joint_names_;
  std::vector<double> latest_action_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr action_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<tienkung_policy_bridge::PolicyBridgeNode>());
  rclcpp::shutdown();
  return 0;
}

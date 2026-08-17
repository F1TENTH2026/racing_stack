// Copyright 2020 F1TENTH Foundation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//   * Neither the name of the {copyright_holder} nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// -*- mode:c++; fill-column: 100; -*-

#include "vesc_ackermann/ackermann_to_vesc.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>

namespace vesc_ackermann
{

using ackermann_msgs::msg::AckermannDriveStamped;
using nav_msgs::msg::Odometry;
using std::placeholders::_1;
using std_msgs::msg::Float64;

namespace
{
// Software hard stop in addition to the vesc_driver brake_max limit. Raising
// this requires an intentional code/config change after physical validation.
// Raised 10 -> 30 A for stronger deceleration; the VESC's own Motor Current Max
// Brake (VESC Tool, not in this repo) still caps whatever is asked for here.
constexpr double kHardBrakeCurrentLimit = 30.0;
}

AckermannToVesc::AckermannToVesc(const rclcpp::NodeOptions & options)
: Node("ackermann_to_vesc_node", options)
{
  // get conversion parameters
  speed_to_erpm_gain_ = declare_parameter<double>("speed_to_erpm_gain");
  speed_to_erpm_offset_ = declare_parameter<double>("speed_to_erpm_offset");
  steering_to_servo_gain_ =
    declare_parameter<double>("steering_angle_to_servo_gain");
  steering_to_servo_offset_ =
    declare_parameter<double>("steering_angle_to_servo_offset");

  // Current-brake controller. The command sent to COMM_SET_CURRENT_BRAKE is a
  // positive motor-phase current; VESC's negative Motor Current Max Brake is
  // the firmware-side limit, not the sign expected on this topic.
  brake_control_enabled_ = declare_parameter<bool>("brake_control_enabled", true);
  brake_current_max_ = declare_parameter<double>("brake_current_max", 10.0);
  brake_current_start_ = declare_parameter<double>("brake_current_start", 5.0);
  brake_current_gain_ = declare_parameter<double>("brake_current_gain", 10.0);
  brake_current_ramp_rate_ = declare_parameter<double>("brake_current_ramp_rate", 100.0);
  brake_engage_speed_error_ =
    declare_parameter<double>("brake_engage_speed_error", 0.15);
  brake_release_speed_error_ =
    declare_parameter<double>("brake_release_speed_error", 0.05);
  brake_release_speed_ = declare_parameter<double>("brake_release_speed", 0.15);
  brake_odom_timeout_ = declare_parameter<double>("brake_odom_timeout", 0.2);

  // Enforce the same safe limits at startup even if an old YAML overrides them.
  brake_current_max_ = std::max(0.0, std::min(brake_current_max_, kHardBrakeCurrentLimit));
  brake_current_start_ = std::max(0.0, std::min(brake_current_start_, brake_current_max_));

  // create publishers to vesc electric-RPM (speed) and servo commands
  erpm_pub_ = create_publisher<Float64>("commands/motor/speed", 10);
  brake_pub_ = create_publisher<Float64>("commands/motor/brake", 10);
  servo_pub_ = create_publisher<Float64>("commands/servo/position", 10);

  // subscribe to ackermann topic
  ackermann_sub_ = create_subscription<AckermannDriveStamped>(
    "ackermann_cmd", 10, std::bind(&AckermannToVesc::ackermannCmdCallback, this, _1));
  odom_sub_ = create_subscription<Odometry>(
    "odom", 10, std::bind(&AckermannToVesc::odomCallback, this, _1));

  // apply gain/offset changes at runtime (ros2 param set, or the vesc_calibration
  // node's apply), so retuning takes effect without restarting this node.
  param_cb_handle_ = add_on_set_parameters_callback(
    std::bind(&AckermannToVesc::onSetParameters, this, _1));
}

rcl_interfaces::msg::SetParametersResult AckermannToVesc::onSetParameters(
  const std::vector<rclcpp::Parameter> & params)
{
  double next_brake_current_max = brake_current_max_;
  double next_brake_current_start = brake_current_start_;
  double next_brake_current_gain = brake_current_gain_;
  double next_brake_current_ramp_rate = brake_current_ramp_rate_;
  double next_brake_engage_speed_error = brake_engage_speed_error_;
  double next_brake_release_speed_error = brake_release_speed_error_;
  double next_brake_release_speed = brake_release_speed_;
  double next_brake_odom_timeout = brake_odom_timeout_;

  for (const auto & p : params) {
    const auto & name = p.get_name();
    if (name == "brake_current_max") {
      next_brake_current_max = p.as_double();
    } else if (name == "brake_current_start") {
      next_brake_current_start = p.as_double();
    } else if (name == "brake_current_gain") {
      next_brake_current_gain = p.as_double();
    } else if (name == "brake_current_ramp_rate") {
      next_brake_current_ramp_rate = p.as_double();
    } else if (name == "brake_engage_speed_error") {
      next_brake_engage_speed_error = p.as_double();
    } else if (name == "brake_release_speed_error") {
      next_brake_release_speed_error = p.as_double();
    } else if (name == "brake_release_speed") {
      next_brake_release_speed = p.as_double();
    } else if (name == "brake_odom_timeout") {
      next_brake_odom_timeout = p.as_double();
    }
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = false;
  if (next_brake_current_max < 0.0 || next_brake_current_max > kHardBrakeCurrentLimit) {
    result.reason = "brake_current_max must be in [0, 30] A";
    return result;
  }
  if (next_brake_current_start < 0.0 ||
    next_brake_current_start > next_brake_current_max)
  {
    result.reason = "brake_current_start must be in [0, brake_current_max] A";
    return result;
  }
  if (next_brake_current_gain < 0.0 || next_brake_current_ramp_rate <= 0.0 ||
    next_brake_engage_speed_error <= next_brake_release_speed_error ||
    next_brake_release_speed_error < 0.0 || next_brake_release_speed < 0.0 ||
    next_brake_odom_timeout <= 0.0)
  {
    result.reason = "invalid brake gain/ramp/hysteresis parameter";
    return result;
  }

  for (const auto & p : params) {
    const auto & name = p.get_name();
    if (name == "speed_to_erpm_gain") {
      speed_to_erpm_gain_ = p.as_double();
    } else if (name == "speed_to_erpm_offset") {
      speed_to_erpm_offset_ = p.as_double();
    } else if (name == "steering_angle_to_servo_gain") {
      steering_to_servo_gain_ = p.as_double();
    } else if (name == "steering_angle_to_servo_offset") {
      steering_to_servo_offset_ = p.as_double();
    } else if (name == "brake_control_enabled") {
      brake_control_enabled_ = p.as_bool();
    } else if (name == "brake_current_max") {
      brake_current_max_ = p.as_double();
    } else if (name == "brake_current_start") {
      brake_current_start_ = p.as_double();
    } else if (name == "brake_current_gain") {
      brake_current_gain_ = p.as_double();
    } else if (name == "brake_current_ramp_rate") {
      brake_current_ramp_rate_ = p.as_double();
    } else if (name == "brake_engage_speed_error") {
      brake_engage_speed_error_ = p.as_double();
    } else if (name == "brake_release_speed_error") {
      brake_release_speed_error_ = p.as_double();
    } else if (name == "brake_release_speed") {
      brake_release_speed_ = p.as_double();
    } else if (name == "brake_odom_timeout") {
      brake_odom_timeout_ = p.as_double();
    }
  }
  result.successful = true;
  return result;
}

void AckermannToVesc::odomCallback(const Odometry::SharedPtr odom)
{
  current_speed_ = odom->twist.twist.linear.x;
  odom_received_ = std::isfinite(current_speed_);
  last_odom_time_ = get_clock()->now();
  odom_time_initialized_ = true;
}

void AckermannToVesc::ackermannCmdCallback(const AckermannDriveStamped::SharedPtr cmd)
{
  // calc steering angle (servo)
  Float64 servo_msg;
  servo_msg.data = steering_to_servo_gain_ * cmd->drive.steering_angle + steering_to_servo_offset_;

  const auto now = get_clock()->now();
  double dt = 0.02;
  if (command_time_initialized_) {
    dt = std::max(0.001, std::min((now - last_command_time_).seconds(), 0.1));
  }
  last_command_time_ = now;
  command_time_initialized_ = true;

  const double target_speed = std::isfinite(cmd->drive.speed) ? cmd->drive.speed : 0.0;
  const double actual_speed_abs = std::abs(current_speed_);
  const double target_speed_abs = std::abs(target_speed);
  const double speed_error = actual_speed_abs - target_speed_abs;
  const bool direction_change =
    actual_speed_abs > brake_release_speed_ && current_speed_ * target_speed < 0.0;
  const bool odom_fresh = odom_received_ && odom_time_initialized_ &&
    (now - last_odom_time_).seconds() <= brake_odom_timeout_;

  bool brake_requested = false;
  if (brake_control_enabled_ && odom_fresh &&
    actual_speed_abs > brake_release_speed_)
  {
    brake_requested = direction_change || speed_error > brake_engage_speed_error_ ||
      (braking_ && speed_error > brake_release_speed_error_);
  }

  if (brake_requested) {
    const double control_error = direction_change ? actual_speed_abs : std::max(speed_error, 0.0);
    const double target_current = std::min(
      brake_current_max_, brake_current_start_ + brake_current_gain_ * control_error);
    const double max_step = brake_current_ramp_rate_ * dt;
    if (target_current > brake_current_) {
      brake_current_ = std::min(target_current, brake_current_ + max_step);
    } else {
      brake_current_ = std::max(target_current, brake_current_ - max_step);
    }

    Float64 brake_msg;
    brake_msg.data = brake_current_;
    brake_pub_->publish(brake_msg);
    if (!braking_) {
      RCLCPP_WARN(
        get_logger(), "Current braking engaged: target=%.2f m/s actual=%.2f m/s",
        target_speed, current_speed_);
    }
    braking_ = true;
  } else {
    // Publishing a speed command switches VESC out of current-brake mode. Do
    // not publish motor/brake in this branch: the last motor command wins.
    Float64 erpm_msg;
    erpm_msg.data = speed_to_erpm_gain_ * target_speed + speed_to_erpm_offset_;
    erpm_pub_->publish(erpm_msg);
    if (braking_) {
      RCLCPP_INFO(
        get_logger(), "Current braking released: target=%.2f m/s actual=%.2f m/s",
        target_speed, current_speed_);
    }
    braking_ = false;
    brake_current_ = 0.0;
  }

  // Steering remains live while longitudinal control is in either mode.
  if (rclcpp::ok()) {
    servo_pub_->publish(servo_msg);
  }
}

}  // namespace vesc_ackermann

#include "rclcpp_components/register_node_macro.hpp"  // NOLINT

RCLCPP_COMPONENTS_REGISTER_NODE(vesc_ackermann::AckermannToVesc)

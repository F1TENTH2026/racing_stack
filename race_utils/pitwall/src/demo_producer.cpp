// Minimal C++ demo: logs a couple of scalar channels + a periodic event so you
// can verify the pitwall -> recorder -> MCAP pipeline end to end.
#include <cmath>

#include <rclcpp/rclcpp.hpp>

#include "pitwall/pitwall.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("pitwall_demo");
  pitwall::init(node.get());

  RCLCPP_INFO(
    node->get_logger(),
    "pitwall_demo: publishing /pitwall/speed,/pitwall/steer (gated on recorder presence)");

  rclcpp::WallRate rate(50.0);
  double t = 0.0;
  int lap = 0;
  while (rclcpp::ok()) {
    pitwall::log("speed", 2.0 + std::sin(t));
    pitwall::log("steer", 0.3 * std::cos(t));
    if (static_cast<int>(t / 5.0) > lap) {
      lap = static_cast<int>(t / 5.0);
      pitwall::event("lap_marker");
    }
    t += 0.02;
    rclcpp::spin_some(node);
    rate.sleep();
  }
  rclcpp::shutdown();
  return 0;
}

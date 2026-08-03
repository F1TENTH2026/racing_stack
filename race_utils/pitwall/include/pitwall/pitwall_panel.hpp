#ifndef PITWALL__PITWALL_PANEL_HPP_
#define PITWALL__PITWALL_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>

class QPushButton;
class QLabel;
class QPlainTextEdit;

namespace pitwall
{

// RViz dockable panel for the UNICORN stack:
//   - State banner (top): colour reflects the state machine state (/state_machine)
//   - Telemetry box: live, terminal-like feed of important events from any node
//     that calls pitwall::event(...) / pitwall.event(...) -> /pitwall/events
//   - Virtual obstacles control (bottom): spawn/remove opponent, clear static
//     obstacles, drive mode / lidar toggles, VP inject seam, opponent speed.
class PitwallPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit PitwallPanel(QWidget * parent = nullptr);
  void onInitialize() override;

Q_SIGNALS:
  // Emitted from the ROS subscription thread; the (auto/queued) connection
  // marshals it onto the Qt GUI thread — touching widgets off-thread is unsafe.
  void stateReceived(const QString & state);
  void telemetryReceived(const QString & line);

private Q_SLOTS:
  void onStateUpdate(const QString & state);
  void onTelemetry(const QString & line);
  void onRemoveOpponent();
  void onClearObstacles();
  void onSpeedUp();
  void onSpeedDown();
  void onModeManual();
  void onModePath();
  void onModeFtg();
  void onToggleEgoLidar();
  void onToggleOppLidar();
  void onSelectOverlay();
  void onSelectMerge();
  void onSelectJoy();
  void onSelectKeyboard();

private:
  void publishSpeedDelta(float delta);
  void publishMode(const char * mode);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr remove_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr clear_obstacles_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ego_lidar_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr opp_lidar_pub_;
  // virtual_perception injection seam selector (overlay XOR merge)
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr inject_mode_pub_;
  // ego human-drive input source selector (joy XOR keyboard) -> simple_mux.
  // Bool: false = joy (existing path), true = keyboard (/joy_keyboard).
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ego_control_pub_;
  // current state machine state -> colour banner (survives removing /state_marker)
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  // pitwall important-events feed -> telemetry box
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr events_sub_;
  QPushButton * ego_lidar_btn_ {nullptr};
  QPushButton * opp_lidar_btn_ {nullptr};
  QPushButton * scan_overlay_btn_ {nullptr};
  QPushButton * tracking_merge_btn_ {nullptr};
  QPushButton * joy_src_btn_ {nullptr};
  QPushButton * keyboard_src_btn_ {nullptr};

  QLabel * state_banner_ {nullptr};          // top: current state, coloured per state
  QPlainTextEdit * telemetry_log_ {nullptr}; // terminal-like live telemetry feed
  QString last_state_;
  QLabel * status_label_ {nullptr};
  double speed_step_ {0.5};
};

}  // namespace pitwall

#endif  // PITWALL__PITWALL_PANEL_HPP_

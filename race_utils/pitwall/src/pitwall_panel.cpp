#include "pitwall/pitwall_panel.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFrame>
#include <QPlainTextEdit>
#include <QFont>
#include <QTime>

#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace pitwall
{

PitwallPanel::PitwallPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout;

  // ===== top: current state, background coloured per state machine state =====
  state_banner_ = new QLabel("STATE: —");
  state_banner_->setAlignment(Qt::AlignCenter);
  state_banner_->setMinimumHeight(34);
  state_banner_->setStyleSheet(
    "background-color:#808080; color:white; font-weight:bold; "
    "border-radius:3px; padding:4px;");
  layout->addWidget(state_banner_);

  // ===== telemetry: terminal-like live feed of pitwall events from any node =====
  layout->addWidget(new QLabel("Telemetry:"));
  telemetry_log_ = new QPlainTextEdit;
  telemetry_log_->setReadOnly(true);
  telemetry_log_->setMaximumBlockCount(500);        // cap memory; old lines drop off
  telemetry_log_->setMinimumHeight(140);
  {
    QFont mono("monospace");
    mono.setStyleHint(QFont::TypeWriter);
    telemetry_log_->setFont(mono);
  }
  layout->addWidget(telemetry_log_, /*stretch=*/1);

  status_label_ = new QLabel("");
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  // ===== Ego control: human-drive input source (joy XOR keyboard) =====
  // Sits beside the telemetry; the selection is a Bool trigger consumed by
  // simple_mux (false = joy / existing path, true = keyboard / /joy_keyboard).
  layout->addWidget(new QLabel("Ego control (pick one):"));
  auto * ego_row = new QHBoxLayout;
  joy_src_btn_ = new QPushButton("Joy");
  keyboard_src_btn_ = new QPushButton("Keyboard");
  joy_src_btn_->setCheckable(true);
  keyboard_src_btn_->setCheckable(true);
  joy_src_btn_->setChecked(true);        // default = joy (matches simple_mux default)
  keyboard_src_btn_->setChecked(false);
  ego_row->addWidget(joy_src_btn_);
  ego_row->addWidget(keyboard_src_btn_);
  layout->addLayout(ego_row);

  // Tiny cheat-sheet for the keyboard teleop (keyboard_joy_node) controls.
  // ASCII only — conda's libfontconfig segfaults on exotic glyphs (arrows, dots).
  auto * ego_hint = new QLabel("Keyboard: arrows drive | H=human | A=auto");
  ego_hint->setStyleSheet("color:gray; font-size:10px;");
  layout->addWidget(ego_hint);

  // Push the controls block to the very bottom of the panel.
  layout->addStretch();

  // ===== bottom: Virtual obstacles control =====
  auto * vobs_header = new QLabel("<b>Virtual obstacles control</b>");
  layout->addWidget(vobs_header);

  auto * hint = new QLabel("Spawn opponent: use the <i>2D Goal Pose</i> tool.");
  hint->setWordWrap(true);
  layout->addWidget(hint);

  auto * obs_row = new QHBoxLayout;
  auto * remove_btn = new QPushButton("Remove opponent");
  auto * clear_obs_btn = new QPushButton("Clear static obstacles");
  obs_row->addWidget(remove_btn);
  obs_row->addWidget(clear_obs_btn);
  layout->addLayout(obs_row);

  auto * mode_row = new QHBoxLayout;
  auto * manual_btn = new QPushButton("Manual");
  auto * path_btn = new QPushButton("Path");
  auto * ftg_btn = new QPushButton("FTG");
  mode_row->addWidget(manual_btn);
  mode_row->addWidget(path_btn);
  mode_row->addWidget(ftg_btn);
  layout->addLayout(mode_row);

  layout->addWidget(new QLabel("LiDAR (off = faster sim):"));
  auto * lidar_row = new QHBoxLayout;
  ego_lidar_btn_ = new QPushButton("Ego LiDAR: ON");
  opp_lidar_btn_ = new QPushButton("Opp LiDAR: ON");
  ego_lidar_btn_->setCheckable(true);
  opp_lidar_btn_->setCheckable(true);
  ego_lidar_btn_->setChecked(true);
  opp_lidar_btn_->setChecked(true);
  lidar_row->addWidget(ego_lidar_btn_);
  lidar_row->addWidget(opp_lidar_btn_);
  layout->addLayout(lidar_row);

  layout->addWidget(new QLabel("Virtual perception inject (pick one):"));
  auto * vp_row = new QHBoxLayout;
  scan_overlay_btn_ = new QPushButton("LiDAR Overlay");
  tracking_merge_btn_ = new QPushButton("Tracking Merge");
  scan_overlay_btn_->setCheckable(true);
  tracking_merge_btn_->setCheckable(true);
  scan_overlay_btn_->setChecked(true);    // default seam = overlay (matches node default)
  tracking_merge_btn_->setChecked(false);
  vp_row->addWidget(scan_overlay_btn_);
  vp_row->addWidget(tracking_merge_btn_);
  layout->addLayout(vp_row);

  auto * speed_row = new QHBoxLayout;
  speed_row->addWidget(new QLabel("Opp speed:"));
  auto * down_btn = new QPushButton("-");
  auto * up_btn = new QPushButton("+");
  down_btn->setMaximumWidth(40);
  up_btn->setMaximumWidth(40);
  speed_row->addWidget(down_btn);
  speed_row->addWidget(up_btn);
  layout->addLayout(speed_row);

  setLayout(layout);

  // ROS callbacks (subscription thread) -> GUI thread via queued connection.
  connect(this, &PitwallPanel::stateReceived, this, &PitwallPanel::onStateUpdate);
  connect(this, &PitwallPanel::telemetryReceived, this, &PitwallPanel::onTelemetry);

  connect(remove_btn, &QPushButton::clicked, this, &PitwallPanel::onRemoveOpponent);
  connect(clear_obs_btn, &QPushButton::clicked, this, &PitwallPanel::onClearObstacles);
  connect(up_btn, &QPushButton::clicked, this, &PitwallPanel::onSpeedUp);
  connect(down_btn, &QPushButton::clicked, this, &PitwallPanel::onSpeedDown);
  connect(manual_btn, &QPushButton::clicked, this, &PitwallPanel::onModeManual);
  connect(path_btn, &QPushButton::clicked, this, &PitwallPanel::onModePath);
  connect(ftg_btn, &QPushButton::clicked, this, &PitwallPanel::onModeFtg);
  connect(ego_lidar_btn_, &QPushButton::clicked, this, &PitwallPanel::onToggleEgoLidar);
  connect(opp_lidar_btn_, &QPushButton::clicked, this, &PitwallPanel::onToggleOppLidar);
  connect(scan_overlay_btn_, &QPushButton::clicked, this, &PitwallPanel::onSelectOverlay);
  connect(tracking_merge_btn_, &QPushButton::clicked, this, &PitwallPanel::onSelectMerge);
  connect(joy_src_btn_, &QPushButton::clicked, this, &PitwallPanel::onSelectJoy);
  connect(keyboard_src_btn_, &QPushButton::clicked, this, &PitwallPanel::onSelectKeyboard);
}

void PitwallPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  remove_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/sim/remove_opponent", 10);
  clear_obstacles_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/sim/clear_obstacles", 10);
  speed_pub_ = node_->create_publisher<std_msgs::msg::Float32>("/sim/opp_speed_delta", 10);
  mode_pub_ = node_->create_publisher<std_msgs::msg::String>("/sim/opp_mode", 10);
  ego_lidar_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/sim/ego_lidar_enable", 10);
  opp_lidar_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/sim/opp_lidar_enable", 10);
  inject_mode_pub_ = node_->create_publisher<std_msgs::msg::String>("/vp/inject_mode", 10);
  ego_control_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/ego/use_keyboard", 10);

  // State machine state (std_msgs/String) -> colour banner. Survives removing
  // the /state_marker visualization.
  state_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/state_machine", rclcpp::QoS(1),
    [this](const std_msgs::msg::String::SharedPtr msg) {
      Q_EMIT stateReceived(QString::fromStdString(msg->data));
    });

  // pitwall important-events feed -> Telemetry box. Any node that calls
  // pitwall::event(...) / pitwall.event(...) publishes here (it is gated on a
  // subscriber, so this panel subscribing is what makes those events flow).
  events_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/pitwall/events", rclcpp::QoS(50),
    [this](const std_msgs::msg::String::SharedPtr msg) {
      Q_EMIT telemetryReceived(QString::fromStdString(msg->data));
    });
}

void PitwallPanel::onStateUpdate(const QString & state)
{
  // Colour map mirrors state_machine.py's /state_marker:
  //   GB_TRACK=blue OVERTAKE=red TRAILING=yellow ATTACK=magenta
  //   FTGONLY=white RECOVERY=green  (START/LOSTLINE/other=grey)
  QString bg = "#808080", fg = "white";
  if (state == "GB_TRACK")      {bg = "#1565c0"; fg = "white";}
  else if (state == "OVERTAKE") {bg = "#d32f2f"; fg = "white";}
  else if (state == "TRAILING") {bg = "#fbc02d"; fg = "black";}
  else if (state == "ATTACK")   {bg = "#c2185b"; fg = "white";}
  else if (state == "FTGONLY")  {bg = "#fafafa"; fg = "black";}
  else if (state == "RECOVERY") {bg = "#388e3c"; fg = "white";}

  if (state_banner_) {
    state_banner_->setText("STATE: " + state);
    state_banner_->setStyleSheet(
      QString("background-color:%1; color:%2; font-weight:bold; "
              "border-radius:3px; padding:4px;").arg(bg, fg));
  }
  last_state_ = state;
}

void PitwallPanel::onTelemetry(const QString & line)
{
  if (telemetry_log_) {
    telemetry_log_->appendPlainText(
      QTime::currentTime().toString("HH:mm:ss.zzz") + "  " + line);
  }
}

void PitwallPanel::onRemoveOpponent()
{
  if (remove_pub_) {
    remove_pub_->publish(std_msgs::msg::Empty());
    if (status_label_) {status_label_->setText("Removed opponent");}
  }
}

void PitwallPanel::onClearObstacles()
{
  if (clear_obstacles_pub_) {
    clear_obstacles_pub_->publish(std_msgs::msg::Empty());
    if (status_label_) {status_label_->setText("Cleared static obstacles");}
  }
}

void PitwallPanel::publishSpeedDelta(float delta)
{
  if (!speed_pub_) {return;}
  std_msgs::msg::Float32 msg;
  msg.data = delta;
  speed_pub_->publish(msg);
  if (status_label_) {
    status_label_->setText(QString("Opp speed %1%2 m/s")
      .arg(delta >= 0 ? "+" : "").arg(static_cast<double>(delta)));
  }
}

void PitwallPanel::onSpeedUp() {publishSpeedDelta(static_cast<float>(speed_step_));}
void PitwallPanel::onSpeedDown() {publishSpeedDelta(static_cast<float>(-speed_step_));}

void PitwallPanel::publishMode(const char * mode)
{
  if (!mode_pub_) {return;}
  std_msgs::msg::String msg;
  msg.data = mode;
  mode_pub_->publish(msg);
  if (status_label_) {status_label_->setText(QString("Opp mode: %1").arg(mode));}
}

void PitwallPanel::onModeManual() {publishMode("manual");}
void PitwallPanel::onModePath() {publishMode("path");}

void PitwallPanel::onModeFtg()
{
  publishMode("ftg");
  // FTG drives off the opponent's lidar -> force it ON (UI + topic).
  if (opp_lidar_btn_) {
    opp_lidar_btn_->setChecked(true);
    opp_lidar_btn_->setText("Opp LiDAR: ON");
  }
  if (opp_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = true;
    opp_lidar_pub_->publish(m);
  }
}

void PitwallPanel::onToggleEgoLidar()
{
  bool on = ego_lidar_btn_->isChecked();
  ego_lidar_btn_->setText(on ? "Ego LiDAR: ON" : "Ego LiDAR: OFF");
  if (ego_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = on;
    ego_lidar_pub_->publish(m);
  }
}

void PitwallPanel::onToggleOppLidar()
{
  bool on = opp_lidar_btn_->isChecked();
  opp_lidar_btn_->setText(on ? "Opp LiDAR: ON" : "Opp LiDAR: OFF");
  if (opp_lidar_pub_) {
    std_msgs::msg::Bool m;
    m.data = on;
    opp_lidar_pub_->publish(m);
  }
}

void PitwallPanel::onSelectOverlay()
{
  // mutually exclusive: choosing the overlay seam deselects merge
  scan_overlay_btn_->setChecked(true);
  tracking_merge_btn_->setChecked(false);
  if (inject_mode_pub_) {
    std_msgs::msg::String m;
    m.data = "overlay";
    inject_mode_pub_->publish(m);
  }
  if (status_label_) {status_label_->setText("VP inject: LiDAR overlay");}
}

void PitwallPanel::onSelectMerge()
{
  scan_overlay_btn_->setChecked(false);
  tracking_merge_btn_->setChecked(true);
  if (inject_mode_pub_) {
    std_msgs::msg::String m;
    m.data = "merge";
    inject_mode_pub_->publish(m);
  }
  if (status_label_) {status_label_->setText("VP inject: tracking merge");}
}

void PitwallPanel::onSelectJoy()
{
  // mutually exclusive: choosing joy deselects keyboard
  joy_src_btn_->setChecked(true);
  keyboard_src_btn_->setChecked(false);
  if (ego_control_pub_) {
    std_msgs::msg::Bool m;
    m.data = false;   // false = joy (existing /joy path)
    ego_control_pub_->publish(m);
  }
  if (status_label_) {status_label_->setText("Ego control: Joy");}
}

void PitwallPanel::onSelectKeyboard()
{
  joy_src_btn_->setChecked(false);
  keyboard_src_btn_->setChecked(true);
  if (ego_control_pub_) {
    std_msgs::msg::Bool m;
    m.data = true;    // true = keyboard (/joy_keyboard path)
    ego_control_pub_->publish(m);
  }
  if (status_label_) {status_label_->setText("Ego control: Keyboard");}
}

}  // namespace pitwall

PLUGINLIB_EXPORT_CLASS(pitwall::PitwallPanel, rviz_common::Panel)

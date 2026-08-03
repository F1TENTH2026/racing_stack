#include "pitwall/pitwall.hpp"

#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <unordered_map>

#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>

namespace pitwall
{
namespace
{

struct Impl
{
  rclcpp::Node * node = nullptr;
  std::shared_ptr<rclcpp::Node> owned;  // only set when lazily self-created
  std::unordered_map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> scalar_pubs;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub;
  std::mutex mtx;

  // Returns a usable node, lazily creating a hidden one if init() was never
  // called. Requires rclcpp::init() to have run; returns nullptr otherwise.
  rclcpp::Node * get_node()
  {
    if (node) {
      return node;
    }
    if (!rclcpp::ok()) {
      return nullptr;
    }
    owned = std::make_shared<rclcpp::Node>("pitwall_" + std::to_string(::getpid()));
    node = owned.get();
    return node;
  }
};

Impl & impl()
{
  static Impl instance;
  return instance;
}

// Topic namespace prefix. Defaults to "/pitwall" (visible in `ros2 topic list`
// and to foxglove_bridge for live viewing). Override with env
// PITWALL_TOPIC_PREFIX -- e.g. "/_pitwall" (leading underscore) makes them ROS
// hidden topics; the recorder passes --include-hidden-topics either way.
const std::string & prefix()
{
  static const std::string p = [] {
    const char * e = std::getenv("PITWALL_TOPIC_PREFIX");
    std::string v = (e && *e) ? e : "/pitwall";
    while (v.size() > 1 && v.back() == '/') {
      v.pop_back();
    }
    return v;
  }();
  return p;
}

// ROS topic names allow only [A-Za-z0-9_/]; map everything else (e.g. the '.'
// in "state.x") to '_' so keys map cleanly to topics.
std::string sanitize(const std::string & key)
{
  std::string s = key;
  for (auto & c : s) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '/')) {
      c = '_';
    }
  }
  return s;
}

}  // namespace

void init(rclcpp::Node * node)
{
  auto & I = impl();
  std::lock_guard<std::mutex> lk(I.mtx);
  I.node = node;
}

void log(const std::string & key, double value)
{
  auto & I = impl();
  std::lock_guard<std::mutex> lk(I.mtx);
  rclcpp::Node * n = I.get_node();
  if (!n) {
    return;
  }
  auto it = I.scalar_pubs.find(key);
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub;
  if (it == I.scalar_pubs.end()) {
    pub = n->create_publisher<std_msgs::msg::Float64>(prefix() + "/" + sanitize(key), rclcpp::QoS(50));
    I.scalar_pubs.emplace(key, pub);
  } else {
    pub = it->second;
  }
  if (pub->get_subscription_count() == 0) {
    return;  // no recorder listening -> no-op
  }
  std_msgs::msg::Float64 m;
  m.data = value;
  pub->publish(m);
}

void event(const std::string & name)
{
  auto & I = impl();
  std::lock_guard<std::mutex> lk(I.mtx);
  rclcpp::Node * n = I.get_node();
  if (!n) {
    return;
  }
  if (!I.event_pub) {
    I.event_pub = n->create_publisher<std_msgs::msg::String>(prefix() + "/events", rclcpp::QoS(50));
  }
  if (I.event_pub->get_subscription_count() == 0) {
    return;
  }
  std_msgs::msg::String m;
  m.data = name;
  I.event_pub->publish(m);
}

}  // namespace pitwall

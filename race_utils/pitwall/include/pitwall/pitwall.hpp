// pitwall — ergonomic, one-line telemetry logging for the UNICORN racing stack.
//
// Call pitwall::log("speed", 2.0) anywhere in any node. Internally it lazily
// creates a publisher on /pitwall/<key> (std_msgs/Float64) and publishes ONLY
// when a recorder (subscriber) is present — otherwise it is a cheap no-op.
// No per-node files, no message definitions, no publisher boilerplate at the
// call site. A recorder node captures /pitwall/* (plus sensor topics) into a
// single MCAP for Foxglove.
#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>

namespace pitwall
{

// Bind pitwall to an existing node (recommended: reuses its DDS participant
// instead of spawning a second one per process). Call once, e.g. in main().
void init(rclcpp::Node * node);

// Log a scalar telemetry value under `key`. Publishes std_msgs/Float64 on
// /pitwall/<sanitized-key>, gated on a recorder being subscribed.
void log(const std::string & key, double value);

// Log a sparse event (mode change, fault, lap marker) on /pitwall/events.
void event(const std::string & name);

}  // namespace pitwall

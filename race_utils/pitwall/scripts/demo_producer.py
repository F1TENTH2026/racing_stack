#!/usr/bin/env python3
"""Minimal Python demo for the pitwall -> recorder -> MCAP pipeline."""

import math

import rclpy

import pitwall


def main():
    rclpy.init()
    node = rclpy.create_node("pitwall_demo_py")
    pitwall.init(node)
    node.get_logger().info(
        "pitwall_demo_py: publishing /pitwall/speed,/pitwall/steer (gated on recorder)")

    state = {"t": 0.0, "lap": 0}

    def tick():
        t = state["t"]
        pitwall.log("speed", 2.0 + math.sin(t))
        pitwall.log("steer", 0.3 * math.cos(t))
        if int(t / 5.0) > state["lap"]:
            state["lap"] = int(t / 5.0)
            pitwall.event("lap_marker")
        state["t"] = t + 0.02

    node.create_timer(0.02, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

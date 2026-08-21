#!/usr/bin/env python3
"""Low-overhead, single-process topic period sampler for the race stack."""
import argparse
import json
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import BehaviorStrategy, ObstacleArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan


TOPICS = (
    ('/vesc/odom', Odometry, False),
    ('/vesc/sensors/imu', Imu, True),
    ('/scan', LaserScan, True),
    ('/pf/pose/odom', Odometry, False),
    ('/odometry/local', Odometry, False),
    ('/car_state/odom', Odometry, False),
    ('/detect/raw_obstacles', ObstacleArray, False),
    ('/tracking/obstacles', ObstacleArray, False),
    ('/behavior_strategy', BehaviorStrategy, False),
    ('/vesc/high_level/ackermann_cmd', AckermannDriveStamped, False),
)


class Sampler(Node):
    def __init__(self):
        super().__init__('stack_topic_sampler')
        self.last = {}
        self.periods = {name: [] for name, _, _ in TOPICS}
        self.subscriptions = []
        for name, msg_type, sensor_qos in TOPICS:
            cb = lambda _msg, topic=name: self.sample(topic)
            qos = qos_profile_sensor_data if sensor_qos else 10
            self.subscriptions.append(self.create_subscription(msg_type, name, cb, qos))

    def sample(self, topic):
        now = time.monotonic()
        previous = self.last.get(topic)
        if previous is not None:
            self.periods[topic].append(now - previous)
        self.last[topic] = now

    def report(self):
        result = {}
        for topic, values in self.periods.items():
            if not values:
                result[topic] = {'samples': 0}
                continue
            result[topic] = {
                'samples': len(values),
                'mean_hz': 1.0 / statistics.mean(values),
                'mean_period_ms': statistics.mean(values) * 1e3,
                'min_period_ms': min(values) * 1e3,
                'max_period_ms': max(values) * 1e3,
                'stddev_period_ms': statistics.pstdev(values) * 1e3,
            }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Sampler()
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
    finally:
        with open(args.output, 'w') as stream:
            json.dump(node.report(), stream, indent=2, sort_keys=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

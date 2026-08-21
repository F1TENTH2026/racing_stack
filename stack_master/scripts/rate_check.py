#!/usr/bin/env python3
"""rate_check.py — one-process replacement for four `ros2 topic hz` terminals.

Subscribes to the stack's four rate-critical topics at once and prints a single
refreshing table. `ros2 topic hz` takes one topic per invocation, so watching
these four means four rclpy processes competing for the very CPU being measured
(see cpu_profile.sh's logs: 3.7% idle, load 13.8 on 6 cores). This is one node
with four subscriptions instead.

What matters here is the MINIMUM and the jitter, not the mean: a topic averaging
40 Hz that drops to 8 Hz for 300 ms is what stops the car (controller_manager.py
zeroes speed after ~250 ms without /local_waypoints). So `min` and `std` are the
columns to read; `mean` alone hides exactly the failure being hunted.

Run it ON THE CAR, not the pitwall laptop -- measuring over WiFi conflates DDS
loss with the on-car scheduling delay this is meant to isolate.

Usage:
    python3 rate_check.py [--window N] [--period SEC] [--log FILE]
"""

import argparse
import statistics
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray

# (topic, type, expected Hz). Expected values come from the rates their inputs
# actually run at: /scan is the UST-10LX's fixed 25 ms cycle, /pf/pose/odom is
# scan-driven, /car_state/odom follows vesc_driver's 20 ms timer, and
# /local_waypoints follows state_machine_params.yaml's `rate`.
TOPICS = [
    ('/scan',            LaserScan, 40.0),
    ('/pf/pose/odom',    Odometry,  40.0),
    ('/car_state/odom',  Odometry,  50.0),
    ('/local_waypoints', WpntArray, 80.0),
]


class RateCheck(Node):
    def __init__(self, window, period, log):
        super().__init__('rate_check')
        self.window = window
        self.log = log
        self.stamps = {t: deque(maxlen=window) for t, _, _ in TOPICS}
        self.counts = {t: 0 for t, _, _ in TOPICS}

        for topic, msg_type, _ in TOPICS:
            # sensor QoS (BEST_EFFORT) matches /scan and the odometry publishers;
            # a RELIABLE subscriber would simply never match them.
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._tick(t),
                qos_profile_sensor_data)

        self.create_timer(period, self._report)
        self.t0 = time.monotonic()

    def _tick(self, topic):
        self.stamps[topic].append(time.monotonic())
        self.counts[topic] += 1

    def _report(self):
        rows = []
        for topic, _, expected in TOPICS:
            s = self.stamps[topic]
            if len(s) < 3:
                rows.append((topic, expected, None, None, None, None, self.counts[topic]))
                continue
            gaps = [b - a for a, b in zip(s, list(s)[1:]) if b > a]
            if not gaps:
                rows.append((topic, expected, None, None, None, None, self.counts[topic]))
                continue
            hz = [1.0 / g for g in gaps]
            rows.append((topic, expected, statistics.mean(hz), min(hz),
                         statistics.pstdev(hz), max(gaps) * 1000.0, self.counts[topic]))

        elapsed = time.monotonic() - self.t0
        out = [f'\n=== rate_check  t={elapsed:6.1f}s  (window={self.window}) ===',
               f'{"topic":<20}{"expect":>7}{"mean":>8}{"min":>8}{"std":>8}{"maxgap":>9}{"msgs":>8}  status']
        for topic, expected, mean, mn, std, maxgap, n in rows:
            if mean is None:
                out.append(f'{topic:<20}{expected:>7.0f}{"--":>8}{"--":>8}{"--":>8}{"--":>9}{n:>8}  NO DATA')
                continue
            # Flag on the minimum, not the mean. A mean at target with a min far
            # below it is precisely the drop-out that trips the controller's
            # waypoint watchdog.
            if mn < expected * 0.5 or maxgap > 250:
                status = 'BAD   <-- 드롭아웃'
            elif mean < expected * 0.9 or std > expected * 0.1:
                status = 'WARN'
            else:
                status = 'ok'
            out.append(f'{topic:<20}{expected:>7.0f}{mean:>8.1f}{mn:>8.1f}{std:>8.1f}'
                       f'{maxgap:>8.0f}ms{n:>8}  {status}')
        text = '\n'.join(out)
        print(text, flush=True)
        if self.log:
            self.log.write(text + '\n')
            self.log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=200,
                    help='messages kept per topic for the statistics (default 200)')
    ap.add_argument('--period', type=float, default=2.0,
                    help='seconds between printed reports (default 2)')
    ap.add_argument('--log', type=argparse.FileType('w'), default=None,
                    help='also append each report to this file')
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = RateCheck(args.window, args.period, args.log)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

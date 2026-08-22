#!/usr/bin/env python3
"""rate_check.py — one-process replacement for four `ros2 topic hz` terminals.

Subscribes to the stack's four rate-critical topics at once and prints a single
refreshing table. `ros2 topic hz` takes one topic per invocation, so watching
these four means four rclpy processes competing for the very CPU being measured
(see cpu_profile.sh's logs: 3.7% idle, load 13.8 on 6 cores). This is one node
with four subscriptions instead.

Read the `avg` column, not `inst`. `avg` is messages/elapsed over the report
window -- the rate the topic actually sustained. `inst` is the mean of the
per-gap 1/dt, which DIVERGES under burst delivery: when a starved subscriber
finally gets CPU, DDS hands it a queued batch back-to-back, the gaps go to
~0, and 1/dt shoots to hundreds of Hz. A first run of this tool printed
/scan at 641 Hz that way while the topic was really doing 27. `inst` is kept
only because that divergence is itself the signature of starvation.

`maxgap` is the number that stops the car: controller_manager.py zeroes speed
after ~250 ms without /local_waypoints.

NOTE the counts are a LOWER BOUND. The subscriptions use sensor QoS (BEST_EFFORT,
depth 5), so messages that arrive while this node is descheduled are dropped and
never counted. Under heavy contention this tool under-reports every topic --
cpu_profile.sh (a `top -b` wrapper, no ROS involved) is the ground truth.

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
        # counts at the previous report, so `avg` covers the window just elapsed
        # rather than smearing a stall across the whole session.
        self.prev_counts = {t: 0 for t, _, _ in TOPICS}
        self.prev_report = time.monotonic()

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
        now = time.monotonic()
        span = max(now - self.prev_report, 1e-6)
        rows = []
        for topic, _, expected in TOPICS:
            avg = (self.counts[topic] - self.prev_counts[topic]) / span
            self.prev_counts[topic] = self.counts[topic]
            s = self.stamps[topic]
            gaps = [b - a for a, b in zip(s, list(s)[1:]) if b > a] if len(s) >= 3 else []
            if not gaps:
                rows.append((topic, expected, avg, None, None, None, self.counts[topic]))
                continue
            hz = [1.0 / g for g in gaps]
            rows.append((topic, expected, avg, statistics.mean(hz), min(hz),
                         max(gaps) * 1000.0, self.counts[topic]))
        self.prev_report = now

        elapsed = now - self.t0
        out = [f'\n=== rate_check  t={elapsed:6.1f}s  (window={self.window}) ===',
               f'{"topic":<20}{"expect":>7}{"avg":>8}{"inst":>8}{"min":>8}{"maxgap":>9}{"msgs":>8}  status']
        for topic, expected, avg, inst, mn, maxgap, n in rows:
            if inst is None:
                out.append(f'{topic:<20}{expected:>7.0f}{avg:>8.1f}{"--":>8}{"--":>8}{"--":>9}{n:>8}  NO DATA')
                continue
            # Flag on the minimum, not the mean. A mean at target with a min far
            # below it is precisely the drop-out that trips the controller's
            # waypoint watchdog.
            # Judge on avg and maxgap. inst is diagnostic only -- an inst far
            # ABOVE expect means burst delivery, i.e. this node was descheduled.
            if avg < expected * 0.5 or maxgap > 250:
                status = 'BAD   <-- 드롭아웃'
            elif avg < expected * 0.9:
                status = 'WARN'
            else:
                status = 'ok'
            if inst > expected * 2:
                status += '  [버스트=기아상태]'
            out.append(f'{topic:<20}{expected:>7.0f}{avg:>8.1f}{inst:>8.1f}{mn:>8.1f}'
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

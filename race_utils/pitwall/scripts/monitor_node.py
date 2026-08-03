#!/usr/bin/env python3
"""pitwall recorder monitor.

Answers "is recording costing us?" by sampling the rosbag2 recorder process's
CPU% and RSS once a second and logging them through pitwall itself -- so the
numbers land on /pitwall/monitor/* and get captured into the SAME MCAP as the
data. In Foxglove you can then overlay recorder cost against algorithm
performance on one timeline.

It finds the recorder by command-line match (default: the `ros2 bag record`
process launched alongside it) via /proc, so there is no dependency on psutil.
"""

import glob
import os
import time

import rclpy
from rclpy.node import Node

import pitwall

_CLK = os.sysconf("SC_CLK_TCK")
_PAGE_MB = os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _find_pids(needle):
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(os.path.join(d, "cmdline"), "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if needle in cmd and "monitor_node" not in cmd:
            pids.append(int(os.path.basename(d)))
    return pids


def _proc_cpu_rss(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            parts = f.read().split()
        cpu = int(parts[13]) + int(parts[14])  # utime + stime (clock ticks)
        with open("/proc/{}/statm".format(pid)) as f:
            rss = int(f.read().split()[1]) * _PAGE_MB
        return cpu, rss
    except (OSError, IndexError, ValueError):
        return None, None


class MonitorNode(Node):
    def __init__(self):
        super().__init__("pitwall_monitor")
        pitwall.init(self)
        self._needle = self.declare_parameter("watch_cmdline", "bag record").value
        self._last = {}  # pid -> (cpu_ticks, monotonic_t)
        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            "pitwall_monitor: watching processes matching '{}'".format(self._needle))

    def _tick(self):
        now = time.monotonic()
        pids = _find_pids(self._needle)
        total_cpu_pct = 0.0
        total_rss = 0.0
        for pid in pids:
            cpu, rss = _proc_cpu_rss(pid)
            if cpu is None:
                continue
            prev = self._last.get(pid)
            self._last[pid] = (cpu, now)
            if prev:
                dt = now - prev[1]
                if dt > 0:
                    total_cpu_pct += 100.0 * (cpu - prev[0]) / _CLK / dt
            total_rss += rss
        pitwall.log("monitor/recorder_cpu_percent", total_cpu_pct)
        pitwall.log("monitor/recorder_rss_mb", total_rss)
        try:
            pitwall.log("monitor/system_load1", os.getloadavg()[0])
        except OSError:
            pass


def main():
    rclpy.init()
    node = MonitorNode()
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

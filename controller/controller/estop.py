import math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class EStop:
    def __init__(self, node):
        self._logger = node.get_logger()
        # iTTC threshold [s]: stop if any front beam will be hit sooner than this.
        try:
            if not node.has_parameter('estop_ttc_threshold'):
                node.declare_parameter('estop_ttc_threshold', 0.35)
            self.ttc_thresh = float(node.get_parameter('estop_ttc_threshold').value)
        except Exception:
            self.ttc_thresh = 0.35
        self._fov = math.radians(70.0)

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()
        if scan is None or odom is None:
            return cmd

        v = odom.twist.twist.linear.x
        if v <= 0.1:                       # not driving forward -> nothing to hit
            return cmd

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        n = ranges.shape[0]
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        # instantaneous TTC: range / range-closing-rate (v*cos(angle))
        closing = np.maximum(v * np.cos(angles), 0.01)
        ittc = ranges / closing
        valid = (np.abs(angles) < self._fov) & np.isfinite(ranges) & (ranges > 0.05)

        if np.any(valid & (ittc < self.ttc_thresh)):
            cmd.drive.speed = 0.0
            self._logger.warn('EStop: TTC below threshold -> stop',
                              throttle_duration_sec=1.0)
        return cmd
    
    
    
    
    
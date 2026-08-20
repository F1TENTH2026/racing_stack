#!/usr/bin/env python3
"""Measure the gyro's yaw-rate NOISE the way the EKF actually sees it — while driving.

WHY NOT JUST USE imu_gyro_bias_check.py
---------------------------------------
That script measures a stationary car, so it reports the sensor's own noise floor
and nothing else. On this car the gyro noise while racing is dominated by chassis
vibration — motor, drivetrain, tyres on a rough floor — which a car sitting still
does not produce at all. Sizing ekf_*.yaml's rejection gate from the stationary
number is how you end up rejecting perfectly good gyro samples: the gate is
n_sigmas wide, and if sigma is too small the gate is too narrow, the EKF stops
correcting, and its covariance grows (the pink ellipse on /car_state/odom in
RViz). Use the stationary number as a FLOOR and this one as the real value.

HOW IT WORKS
------------
There is no independent yaw-rate reference on this car to difference against, so
this separates signal from noise by BANDWIDTH instead. The car's real yaw rate is
a vehicle-dynamics signal: smooth, and well under `cutoff_hz`. Vibration noise is
much faster. So the true rate is estimated by smoothing, and what is left over is
taken as noise.

The smoother is a centred Savitzky-Golay (local polynomial fit), not a moving
average. Both are zero-phase — a causal filter would smear the signal into the
residual — but a moving average only cancels a locally CONSTANT signal, so a
turning car leaks its real yaw rate into the residual and inflates the answer.
Measured against a synthetic 1 rad/s, 0.5 Hz signal, that leakage added a roughly
fixed ~1.2e-4 (rad/s)^2, which is +2% when the true noise is sigma 0.05 but +20%
at sigma 0.02. Fitting a local quadratic instead removes the turn itself.

Whatever the kernel k, subtracting it from white noise of variance s^2 leaves
    var(residual) = s^2 * (1 - 2*k[centre] + sum(k^2))
so the residual is divided by that factor to recover the input variance. (For a
moving average of length N that reduces to the familiar 1 - 1/N.)

Any real yaw dynamics still above the cutoff inflate the estimate, which errs
toward a larger sigma — a wider gate and a more forgiving filter. That is the
safe direction to be wrong in.

USAGE
-----
Drive the car normally — a few laps at racing speed, on the surface you race on.
Standstill samples are excluded (see min_speed), so it is fine to start it before
you set off.

    ros2 run stack_master imu_gyro_noise_check.py
    ros2 run stack_master imu_gyro_noise_check.py --ros-args -p duration:=120.0

Or record on the car and analyse the bag afterwards — one command, no playback:

    ros2 bag record -o bags/gyro /vesc/sensors/imu_corrected /vesc/odom  # driving
    ros2 run stack_master imu_gyro_noise_check.py --ros-args -p bag:=bags/gyro

Prefer the bag over `ros2 bag play`: reading it directly uses the RECORDED
timestamps, so the sample rate — and therefore the cutoff — is the real one.
Played back at anything other than rate 1.0 the live path would measure the
playback rate instead and silently move the cutoff.

A bag that contains both driving and standstill (park the car for the last 20 s)
gets you both numbers from one run: the report splits them, and the ratio is what
tells you whether vibration dominates on your surface.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def _savgol_kernel(win, order):
    """Centre-point Savitzky-Golay smoothing kernel: fit a degree-`order`
    polynomial to each `win`-sample window by least squares and keep its value at
    the centre. Symmetric, so it is zero-phase."""
    half = win // 2
    x = np.arange(-half, half + 1, dtype=float)
    vander = np.vander(x, order + 1, increasing=True)
    # Row 0 of the pseudo-inverse maps the window onto the fitted constant term,
    # which — with x centred on 0 — is the fit evaluated at the centre sample.
    return np.linalg.pinv(vander)[0]


class ImuGyroNoiseCheck(Node):
    def __init__(self):
        super().__init__('imu_gyro_noise_check')
        # The CORRECTED topic by default: that is what ekf_*.yaml's imu0 and
        # pf.yaml subscribe to, so it is the signal whose variance they need.
        self.declare_parameter('imu_topic', '/vesc/sensors/imu_corrected')
        self.declare_parameter('odom_topic', '/vesc/odom')
        self.declare_parameter('duration', 60.0)
        # [Hz] Everything above this is treated as noise. The car's yaw rate is
        # steering-bandwidth limited and sits well below it; raise it only if you
        # can show real yaw dynamics are being eaten.
        self.declare_parameter('cutoff_hz', 5.0)
        # Savitzky-Golay polynomial order. 2 cancels the local curvature of a
        # turn; 0 would degenerate to a plain moving average.
        self.declare_parameter('poly_order', 2)
        # [m/s] Samples slower than this are dropped: a stopped car contributes
        # the bench noise floor and would drag the estimate down. 0 disables the
        # gate (and then no odom subscription is needed).
        self.declare_parameter('min_speed', 0.5)
        # Path to a rosbag2 directory. Set it and nothing is subscribed at all:
        # the bag is read straight through and analysed. Empty = live capture.
        self.declare_parameter('bag', '')

        self.duration = float(self.get_parameter('duration').value)
        self.cutoff_hz = float(self.get_parameter('cutoff_hz').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.poly_order = int(self.get_parameter('poly_order').value)
        self.bag = str(self.get_parameter('bag').value)

        self.t = []
        self.wz = []
        self.moving = []
        self.speed = 0.0
        self.t_first = None
        self.done = False

        imu_topic = self.get_parameter('imu_topic').value
        if self.bag:
            self.sub = None
            self.create_timer(0.0, self._run_bag_once)
            self.get_logger().info(f'reading {self.bag} (no subscriptions)')
            return

        self.sub = self.create_subscription(
            Imu, imu_topic, self.imu_cb, qos_profile_sensor_data)
        if self.min_speed > 0.0:
            odom_topic = self.get_parameter('odom_topic').value
            self.odom_sub = self.create_subscription(
                Odometry, odom_topic, self.odom_cb, 10)
            gate = f'samples below {self.min_speed:.2f} m/s ({odom_topic}) are dropped'
        else:
            gate = 'speed gate disabled — every sample counts'
        self.get_logger().info(
            f'DRIVE THE CAR NORMALLY — collecting {self.duration:.0f}s of {imu_topic}; {gate}')

    def _run_bag_once(self):
        """Read the whole bag, then analyse. Speed is INTERPOLATED onto the IMU
        stamps rather than 'whatever odom arrived last', which the live path has
        to settle for — the two streams run at different rates (100 vs 50 Hz)."""
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

        imu_topic = self.get_parameter('imu_topic').value
        odom_topic = self.get_parameter('odom_topic').value

        reader = SequentialReader()
        # storage_id '' lets rosbag2 pick the plugin from the bag's metadata,
        # so this reads mcap and sqlite3 alike.
        reader.open(StorageOptions(uri=self.bag, storage_id=''),
                    ConverterOptions('', ''))

        it, wz, ot, vx = [], [], [], []
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            if topic == imu_topic:
                it.append(stamp * 1e-9)
                wz.append(deserialize_message(data, Imu).angular_velocity.z)
            elif topic == odom_topic:
                ot.append(stamp * 1e-9)
                vx.append(deserialize_message(data, Odometry).twist.twist.linear.x)

        if len(it) < 200:
            self.get_logger().error(
                f'{len(it)} messages on {imu_topic} in {self.bag} — wrong topic or empty bag?')
            raise SystemExit(1)

        self.t = list(np.asarray(it) - it[0])
        self.wz = wz
        if ot:
            speed = np.interp(np.asarray(it), np.asarray(ot), np.abs(np.asarray(vx)))
        else:
            self.get_logger().warn(
                f'no {odom_topic} in the bag — speed gate disabled, standstill included')
            speed = np.full(len(it), np.inf)
        self.speed_series = speed
        self.moving = list(speed >= self.min_speed) if self.min_speed > 0 else [True] * len(it)
        self.get_logger().info(
            f'{len(it)} IMU + {len(ot)} odom messages over {self.t[-1]:.1f}s')
        self.finish()

    def odom_cb(self, msg: Odometry):
        self.speed = abs(msg.twist.twist.linear.x)

    def imu_cb(self, msg: Imu):
        if self.done:
            return
        now = self.get_clock().now()
        if self.t_first is None:
            self.t_first = now
        self.t.append((now - self.t_first).nanoseconds * 1e-9)
        self.wz.append(msg.angular_velocity.z)
        self.moving.append(self.min_speed <= 0.0 or self.speed >= self.min_speed)
        if self.t[-1] >= self.duration:
            self.done = True
            self.finish()

    def finish(self):
        if self.sub is not None:
            self.destroy_subscription(self.sub)
        t = np.asarray(self.t)
        w = np.asarray(self.wz)
        moving = np.asarray(self.moving, dtype=bool)
        n = w.size

        if n < 200:
            self.get_logger().error(f'only {n} samples — is the IMU publishing?')
            raise SystemExit(1)

        # Actual rate, not the nominal 100 Hz: a dropped-sample stream would size
        # the window wrong and silently change what "5 Hz" means.
        dt = np.diff(t)
        rate = 1.0 / float(np.median(dt))
        win = int(round(rate / self.cutoff_hz))
        win = max(3, win + (win + 1) % 2)          # odd, >= 3

        if n < 5 * win:
            self.get_logger().error(
                f'{n} samples is too few for a {win}-sample window; raise duration')
            raise SystemExit(1)

        # Centred (zero-phase) smoothing over the CONTIGUOUS series, then the
        # moving mask is applied to the residual. Filtering per-segment instead
        # would put a transient at every boundary.
        half = win // 2
        kernel = _savgol_kernel(win, min(self.poly_order, win - 1))
        # The kernel is symmetric, so np.convolve's flip is a no-op here.
        smooth = np.convolve(w, kernel, mode='valid')
        resid = w[half:n - half] - smooth
        mask = moving[half:n - half]
        n_moving = int(mask.sum())

        frac = n_moving / max(1, mask.size)
        if n_moving < 5 * win:
            self.get_logger().error(
                f'only {n_moving} moving samples ({frac:.0%}) — drive during the run, '
                'or lower min_speed')
            raise SystemExit(1)

        r = resid[mask]
        # Undo the smoother's own attenuation of white noise so the number is the
        # input variance, not the residual's (see the module docstring).
        attenuation = 1.0 - 2.0 * kernel[half] + float(np.sum(kernel ** 2))
        var = float(r.var(ddof=1)) / attenuation
        sigma = math.sqrt(var)
        raw_var = float(w[half:n - half][mask].var(ddof=1))

        # The standstill half of the same run, when there is one. This is the
        # number a stationary bench measurement would have given you, and the
        # ratio to the driving figure IS the vibration — the thing that makes a
        # stationary measurement the wrong tool for sizing a rejection gate.
        still_var = None
        speeds = getattr(self, 'speed_series', None)
        if speeds is not None and self.min_speed > 0.0:
            still_mask = np.asarray(speeds)[half:n - half] < 0.05
            if int(still_mask.sum()) >= 5 * win:
                still_var = float(resid[still_mask].var(ddof=1)) / (
                    1.0 - 2.0 * kernel[half] + float(np.sum(kernel ** 2)))

        self.get_logger().info(
            f'{n} samples @ {rate:.1f} Hz, {n_moving} moving ({frac:.0%}); '
            f'window {win} samples = {rate / win:.1f} Hz cutoff, '
            f'Savitzky-Golay order {min(self.poly_order, win - 1)}')
        self.get_logger().info(
            f'raw variance (signal INCLUDED, do not use): {raw_var:.3e} (rad/s)^2')
        self.get_logger().info(
            f'NOISE variance: {var:.3e} (rad/s)^2   sigma {sigma:.5f} rad/s '
            f'({math.degrees(sigma):.3f} deg/s)')
        if still_var:
            self.get_logger().info(
                f'  standstill in the same run: {still_var:.3e} (rad/s)^2  '
                f'sigma {math.sqrt(still_var):.5f} rad/s  '
                f'-> driving is {var / still_var:.1f}x the variance '
                f'({math.sqrt(var / still_var):.1f}x the sigma). That gap is vibration; '
                'it is why a stationary measurement cannot size the gate.')

        print('\n# measured WHILE DRIVING — paste into '
              'stack_master/config/CAR/vehicle_config.yaml')
        print('imu_bias_corrector:')
        print('  ros__parameters:')
        print(f'    gyro_variance: [{var:.3e}, {var:.3e}, {var:.3e}]  '
              f'# (rad/s)^2, measured driving @ {self.cutoff_hz:.1f} Hz cutoff')
        print('\n# and keep the vesc/** section in step:')
        print(f'    yaw_rate_variance_gyro: {var:.3e}\n')
        print('# x and y take the z value: only z is fused in this 2D stack, and a\n'
              '# per-axis split measured this way would just be three noisier\n'
              '# estimates of the same vibration.\n')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = ImuGyroNoiseCheck()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

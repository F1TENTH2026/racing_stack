#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import SetParameters
from f110_msgs.msg import LapData


class RacePhaseSpeedNode(Node):
    """
    Raises the global speed cap once the car has completed enough laps.

    Quali runs in three phases (3 clean laps, 3 laps with static obstacles, then a
    2-minute attack) and only the last one is about lap time, so the run starts
    capped and opens up later. That cap is speed_sector_tuner's `global_limit`,
    which clips every sector's scaling; quali.launch.xml seeds it with the slow
    value at launch and this node performs the single upgrade to the fast value.

    Laps come from lap_analyser's /lap_data, whose lap_count is the number of
    COMPLETED laps (it counts from the car's first finish-line crossing, not from
    launch -- so `fast_after_laps` means laps driven, not laps since boot).

    The upgrade latches: once applied it is never undone, and lap_analyser
    restarting its count (/start_log) cannot drop the car back to the slow cap
    mid-attack. Manual overrides still work at any time, this node only ever
    writes `global_limit` once:

        ros2 param set /speed_sector_tuner global_limit 0.5

    All knobs are launch args on quali.launch.xml (speed_scale,
    speed_scale_fast, fast_after_laps); set auto_speed_up:=false to not run
    this node at all and drive the whole session on `speed_scale`.
    """

    def __init__(self):
        super().__init__('race_phase_speed')

        self.declare_parameter('slow_scale', 0.7)
        self.declare_parameter('fast_scale', 1.0)
        self.declare_parameter('fast_after_laps', 10)
        self.declare_parameter('target_node', '/speed_sector_tuner')
        self.declare_parameter('param_name', 'global_limit')

        self.slow_scale = float(self.get_parameter('slow_scale').value)
        self.fast_scale = float(self.get_parameter('fast_scale').value)
        self.fast_after_laps = int(self.get_parameter('fast_after_laps').value)
        self.target_node = self.get_parameter('target_node').value
        self.param_name = self.get_parameter('param_name').value

        self.upgraded = False
        self.last_lap_count = -1

        self.cli = self.create_client(
            SetParameters, f'{self.target_node}/set_parameters')

        self.create_subscription(LapData, '/lap_data', self.lap_cb, QoSProfile(depth=10))

        self.get_logger().info(
            f"[race_phase_speed] {self.param_name}={self.slow_scale} until lap "
            f"{self.fast_after_laps}, then {self.fast_scale} "
            f"(target {self.target_node})"
        )

    def lap_cb(self, msg: LapData):
        self.last_lap_count = msg.lap_count
        if self.upgraded:
            return
        if msg.lap_count < self.fast_after_laps:
            self.get_logger().info(
                f"[race_phase_speed] lap {msg.lap_count}/{self.fast_after_laps} "
                f"at {self.param_name}={self.slow_scale}"
            )
            return
        self._apply(self.fast_scale)

    def _apply(self, value: float):
        # Latch BEFORE the call: a service that is slow or momentarily unavailable
        # must not make the next lap message fire a second upgrade. A failed call
        # is reported as an error the driver can act on (the manual `ros2 param
        # set` above), which beats retrying into an unbounded queue mid-race.
        self.upgraded = True
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"[race_phase_speed] {self.target_node}/set_parameters unavailable; "
                f"{self.param_name} stays at {self.slow_scale}. Set it by hand: "
                f"ros2 param set {self.target_node} {self.param_name} {value}"
            )
            return

        req = SetParameters.Request()
        req.parameters = [
            Parameter(self.param_name, Parameter.Type.DOUBLE, value).to_parameter_msg()
        ]
        future = self.cli.call_async(req)
        future.add_done_callback(lambda f: self._on_set_done(f, value))

    def _on_set_done(self, future, value: float):
        try:
            results = future.result().results
        except Exception as e:  # noqa: BLE001 - never let a service failure kill the race
            self.get_logger().error(f"[race_phase_speed] set_parameters failed: {e}")
            return
        if results and not results[0].successful:
            self.get_logger().error(
                f"[race_phase_speed] {self.target_node} rejected "
                f"{self.param_name}={value}: {results[0].reason}"
            )
            return
        self.get_logger().warn(
            f"[race_phase_speed] lap {self.last_lap_count} reached -> "
            f"{self.param_name} {self.slow_scale} -> {value}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RacePhaseSpeedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

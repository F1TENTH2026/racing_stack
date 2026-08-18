#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import SetParameters
from f110_msgs.msg import LapData


class RacePhaseSpeedNode(Node):
    """
    Hands the car over from a flat opening speed to the map's sector speeds once it
    has completed enough laps.

    Quali runs in three phases (3 clean laps, 3 laps with static obstacles, then a
    2-minute attack) and only the last one is about lap time, so the run opens on
    one conservative speed everywhere and switches to the tuned sectors later:

        opening laps : speed_sector_tuner runs `use_sector_scaling: False`, i.e.
                       a flat `default_scaling` (quali's `speed_scale`) over the
                       whole lap -- the map's sector boundaries do nothing.
        after N laps : this node sets `use_sector_scaling: True` and the map's
                       speed_scaling.yaml takes over AS WRITTEN. Nothing here
                       scales those numbers; what the yaml says is what the car
                       drives. (The tuner's `phase_multiplier` is left alone at
                       its yaml value for manual tuning.)

    Laps come from lap_analyser's /lap_data, whose lap_count is the number of
    COMPLETED laps (it counts from the car's first finish-line crossing, not from
    launch -- so `fast_after_laps` means laps driven, not laps since boot).

    The switch latches: once applied it is never undone, and lap_analyser
    restarting its count (/start_log) cannot drop the car back to the flat opening
    speed mid-attack. Manual overrides still work at any time, this node only ever
    writes the one parameter once:

        ros2 param set /speed_sector_tuner default_scaling 0.5      # opening laps
        ros2 param set /speed_sector_tuner use_sector_scaling true  # switch early
        ros2 param set /speed_sector_tuner phase_multiplier 0.9     # trim sectors

    The knobs are launch args on quali.launch.xml (speed_scale, fast_after_laps);
    set auto_speed_up:=false to not run this node at all and drive the map's
    sectors from the very first lap.
    """

    def __init__(self):
        super().__init__('race_phase_speed')

        self.declare_parameter('default_scale', 0.7)
        self.declare_parameter('fast_after_laps', 10)
        self.declare_parameter('target_node', '/speed_sector_tuner')
        self.declare_parameter('sector_param_name', 'use_sector_scaling')

        # Only logged: the opening-lap speed is seeded by the launch file as the
        # tuner's default_scaling, this node never writes it.
        self.default_scale = float(self.get_parameter('default_scale').value)
        self.fast_after_laps = int(self.get_parameter('fast_after_laps').value)
        self.target_node = self.get_parameter('target_node').value
        self.sector_param_name = self.get_parameter('sector_param_name').value

        self.upgraded = False
        self.last_lap_count = -1

        self.cli = self.create_client(
            SetParameters, f'{self.target_node}/set_parameters')

        self.create_subscription(LapData, '/lap_data', self.lap_cb, QoSProfile(depth=10))

        self.get_logger().info(
            f"[race_phase_speed] flat {self.default_scale} (no sectors) until lap "
            f"{self.fast_after_laps}, then the map's speed_scaling.yaml "
            f"(target {self.target_node})"
        )

    def lap_cb(self, msg: LapData):
        self.last_lap_count = msg.lap_count
        if self.upgraded:
            return
        if msg.lap_count < self.fast_after_laps:
            self.get_logger().info(
                f"[race_phase_speed] lap {msg.lap_count}/{self.fast_after_laps} "
                f"at flat {self.default_scale} (sectors off)"
            )
            return
        self._apply()

    def _apply(self):
        # Latch BEFORE the call: a service that is slow or momentarily unavailable
        # must not make the next lap message fire a second upgrade. A failed call
        # is reported as an error the driver can act on (the manual `ros2 param
        # set` above), which beats retrying into an unbounded queue mid-race.
        self.upgraded = True
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"[race_phase_speed] {self.target_node}/set_parameters unavailable; "
                f"the car stays on the flat {self.default_scale}. Set it by hand: "
                f"ros2 param set {self.target_node} {self.sector_param_name} true"
            )
            return

        req = SetParameters.Request()
        req.parameters = [
            Parameter(self.sector_param_name, Parameter.Type.BOOL, True).to_parameter_msg()
        ]
        future = self.cli.call_async(req)
        future.add_done_callback(self._on_set_done)

    def _on_set_done(self, future):
        try:
            results = future.result().results
        except Exception as e:  # noqa: BLE001 - never let a service failure kill the race
            self.get_logger().error(f"[race_phase_speed] set_parameters failed: {e}")
            return
        rejected = [r.reason for r in results if not r.successful]
        if rejected:
            self.get_logger().error(
                f"[race_phase_speed] {self.target_node} rejected "
                f"{self.sector_param_name}=true: {'; '.join(rejected)}"
            )
            return
        self.get_logger().warn(
            f"[race_phase_speed] lap {self.last_lap_count} reached -> sector "
            f"scaling ON (was flat {self.default_scale})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RacePhaseSpeedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

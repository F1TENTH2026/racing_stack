#!/usr/bin/env python3
"""
Constant-speed joystick teleop for mapping runs.

Wraps the raw pad (/joy) into a restricted Joy stream (/joy_mapping) that
simple_mux consumes while mapping.launch.xml is up, so a mapping lap can only
ever be a slow, forward-only, human-driven lap:

  LT (held)      Drive forward at the fixed mapping speed (mapping.launch.xml's
                 `mapping_speed`, default 1.0 m/s). Released -> 0 m/s.
  Right stick X  Steering, passed straight through to simple_mux.
  everything else is dropped - reverse is impossible (the throttle axis is never
  negative) and autodrive can never be engaged (buttons[5] is forced to 0).

The Joy message this publishes follows the layout simple_mux._handle_joy reads:
axes[1] = throttle (normalised, scaled by the mux's joy_max_speed - which
mapping.launch.xml pins to mapping_speed, so axes[1]=1.0 means exactly that
speed), axes[3] = steering, buttons[4] = humandrive, buttons[5] = autodrive.

buttons[4] is held at 1 unconditionally, not only while LT is down. The mux
latches its last human_drive command and keeps republishing it for
joy_freshness_threshold (1 s); if this node went quiet on LT release the car
would coast on the stale 1.0 m/s command for that whole second. Publishing a
fresh zero-speed humandrive frame at rate_hz instead stops the car immediately.
The same reasoning drives the joy watchdog: if the pad stops publishing (node
killed, receiver lost), this keeps emitting zero-speed frames rather than
letting the mux ride out its freshness window.

The `joy` node itself is NOT started here - launch it separately, e.g.
  ros2 run joy joy_node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class MappingJoyNode(Node):

    def __init__(self):
        super().__init__('mapping_joy')

        self.declare_parameter('joy_in_topic',  '/joy')
        self.declare_parameter('joy_out_topic', '/joy_mapping')
        self.declare_parameter('rate_hz', 50.0)
        # Xbox layout on the Linux joy node: axes[2] = LT, axes[3] = right stick X.
        # An untouched trigger reads 0.0 until it is first moved (it only starts
        # reporting the 1.0 -> -1.0 range afterwards), so "pressed" has to be a
        # test against a negative threshold - 0.0 must read as released.
        self.declare_parameter('trigger_axis', 2)
        self.declare_parameter('trigger_axis_pressed_below', -0.5)
        # Pads that report LT as a digital button instead: set this to the button
        # index and it takes priority over trigger_axis. -1 = use the axis.
        self.declare_parameter('trigger_button', -1)
        self.declare_parameter('steer_axis', 3)
        self.declare_parameter('steer_deadzone', 0.05)
        # Zero the throttle if the pad goes silent for this long.
        self.declare_parameter('joy_timeout', 0.5)
        p = lambda name: self.get_parameter(name).value

        self.trigger_axis      = int(p('trigger_axis'))
        self.trigger_threshold = float(p('trigger_axis_pressed_below'))
        self.trigger_button    = int(p('trigger_button'))
        self.steer_axis        = int(p('steer_axis'))
        self.steer_deadzone    = float(p('steer_deadzone'))
        self.joy_timeout       = float(p('joy_timeout'))

        self.last_joy      = None
        self.last_joy_time = None
        self._warned_stale = False

        self.create_subscription(Joy, p('joy_in_topic'), self._joy_cb, 10)
        self.pub = self.create_publisher(Joy, p('joy_out_topic'), 10)
        self.create_timer(1.0 / float(p('rate_hz')), self._loop)

        src = (f"buttons[{self.trigger_button}]" if self.trigger_button >= 0
               else f"axes[{self.trigger_axis}] < {self.trigger_threshold}")
        self.get_logger().info(
            f"mapping teleop: hold LT ({src}) to drive forward at the fixed mapping "
            f"speed, steer with axes[{self.steer_axis}]. Reverse and autodrive are "
            f"disabled. {p('joy_in_topic')} -> {p('joy_out_topic')}")

    def _joy_cb(self, msg):
        self.last_joy      = msg
        self.last_joy_time = self.get_clock().now()

    def _joy_is_fresh(self):
        if self.last_joy is None:
            return False
        dt = (self.get_clock().now() - self.last_joy_time).nanoseconds / 1e9
        return dt < self.joy_timeout

    def _trigger_pressed(self, msg):
        if self.trigger_button >= 0:
            if len(msg.buttons) <= self.trigger_button:
                return False
            return bool(msg.buttons[self.trigger_button])
        if len(msg.axes) <= self.trigger_axis:
            return False
        return msg.axes[self.trigger_axis] < self.trigger_threshold

    def _steering(self, msg):
        if len(msg.axes) <= self.steer_axis:
            return 0.0
        steer = msg.axes[self.steer_axis]
        return 0.0 if abs(steer) < self.steer_deadzone else steer

    def _loop(self):
        if self._joy_is_fresh():
            self._warned_stale = False
            # forward only: 1.0 == mapping_speed after the mux scales by
            # joy_max_speed, and the throttle axis never goes below 0.
            speed = 1.0 if self._trigger_pressed(self.last_joy) else 0.0
            steer = self._steering(self.last_joy)
        else:
            if self.last_joy is not None and not self._warned_stale:
                self.get_logger().warn('joy stale - holding the car at zero speed')
                self._warned_stale = True
            speed = 0.0
            steer = 0.0

        out = Joy()
        out.header.stamp = self.get_clock().now().to_msg()
        out.axes = [0.0, speed, 0.0, steer]
        # buttons[4] = humandrive (always, see module docstring),
        # buttons[5] = autodrive (never - no autonomy during a mapping lap).
        out.buttons = [0, 0, 0, 0, 1, 0]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MappingJoyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

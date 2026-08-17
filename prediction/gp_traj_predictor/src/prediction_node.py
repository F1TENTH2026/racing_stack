"""Shared ROS 2 node utilities for the synchronous prediction algorithms."""

import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class PredictionNode(Node):
    """A real rclpy Node with a blocking message helper for legacy algorithms.

    The algorithms process one trajectory sample at a time.  Keeping that
    control flow is intentional; while they wait, this method spins this ROS 2
    node so all subscriptions and services continue to be serviced.

    Also carries the debug gate shared by the three gp_traj_predictor nodes, so
    they honour race.launch.xml's debug:=on|off without each repeating it.
    """

    def __init__(self, node_name, **kwargs):
        super().__init__(node_name, **kwargs)
        # race.launch.xml 의 debug:=on|off 에서 내려온다. off 면 마커를 만들지 않는다.
        self.declare_parameter("debug", True)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

    def viz_ok(self, *pubs) -> bool:
        """마커를 만들어도 되는가.

        debug:=off 이면 무조건 False. 켜져 있어도 구독자가 없으면 False —
        pitwall(RViz)은 조종수 랩탑에서 뜨므로, 붙어 있지 않은 동안 Marker 를
        만들어 직렬화하는 것은 순수한 낭비다. particle_filter 가 이미 쓰는 관례.
        """
        if not getattr(self, "debug", True):
            return False
        return any(p.get_subscription_count() > 0 for p in pubs)

    def wait_for_message(self, topic, message_type, timeout_sec=None):
        messages = []
        subscription = self.create_subscription(
            message_type, topic, messages.append, 10)
        start = time.monotonic()
        # A short-lived local executor avoids retaining this node in rclpy's
        # global executor.  The main prediction loop can then use its own
        # continuously spinning executor for subscription callbacks.
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        try:
            while rclpy.ok() and not messages:
                executor.spin_once(timeout_sec=0.1)
                if timeout_sec is not None and time.monotonic() - start >= timeout_sec:
                    raise TimeoutError(f'Timed out waiting for {topic}')
        finally:
            executor.remove_node(self)
            self.destroy_subscription(subscription)
        return messages[0] if messages else None

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

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
    """

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

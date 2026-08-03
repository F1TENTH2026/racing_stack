"""pitwall — ergonomic, one-line telemetry logging for the UNICORN racing stack.

Call ``pitwall.log("speed", 2.0)`` anywhere in any node. Internally it lazily
creates a publisher on ``/pitwall/<key>`` (``std_msgs/Float64``) and publishes
ONLY when a recorder (subscriber) is present -- otherwise it is a cheap no-op.
No per-node files, no message definitions, no publisher boilerplate at the call
site. A recorder node captures ``/pitwall/*`` (plus sensor topics) into a single
MCAP for Foxglove.
"""

import os

import rclpy
from std_msgs.msg import Float64, String

_node = None
_scalar_pubs = {}
_event_pub = None

# Topic namespace prefix. Defaults to "/pitwall" (visible in `ros2 topic list`
# and to foxglove_bridge for live viewing). Override with env
# PITWALL_TOPIC_PREFIX -- e.g. "/_pitwall" (leading underscore) makes them ROS
# hidden topics; the recorder passes --include-hidden-topics either way.
_PREFIX = (os.environ.get("PITWALL_TOPIC_PREFIX") or "/pitwall").rstrip("/") or "/pitwall"


def _sanitize(key):
    # ROS topic names allow only [A-Za-z0-9_/]; map everything else to '_'.
    return "".join(c if (c.isalnum() or c in "_/") else "_" for c in key)


def init(node):
    """Bind pitwall to an existing rclpy node (recommended). Call once."""
    global _node
    _node = node


def _get_node():
    global _node
    if _node is not None:
        return _node
    if not rclpy.ok():
        return None
    _node = rclpy.create_node("pitwall_{}".format(os.getpid()))
    return _node


def log(key, value):
    """Log a scalar telemetry value under ``key`` (no-op if no recorder)."""
    node = _get_node()
    if node is None:
        return
    pub = _scalar_pubs.get(key)
    if pub is None:
        pub = node.create_publisher(Float64, _PREFIX + "/" + _sanitize(key), 50)
        _scalar_pubs[key] = pub
    if pub.get_subscription_count() == 0:
        return
    msg = Float64()
    msg.data = float(value)
    pub.publish(msg)


def event(name):
    """Log a sparse event on ``/pitwall/events`` (no-op if no recorder)."""
    global _event_pub
    node = _get_node()
    if node is None:
        return
    if _event_pub is None:
        _event_pub = node.create_publisher(String, _PREFIX + "/events", 50)
    if _event_pub.get_subscription_count() == 0:
        return
    msg = String()
    msg.data = str(name)
    _event_pub.publish(msg)

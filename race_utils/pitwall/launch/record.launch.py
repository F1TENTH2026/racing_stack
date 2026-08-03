"""pitwall recorder (+ optional live Foxglove bridge).

  ros2 launch pitwall record.launch.py output_dir:=/home/js/runs/lap03
  ros2 launch pitwall record.launch.py live:=true          # record AND stream live

Captures all /pitwall/* telemetry (plus any extra topics matched by `topic_regex`)
into a single MCAP via `ros2 bag record`. On Ctrl-C, launch sends SIGINT to the
recorder, which finalizes the MCAP cleanly. A monitor node samples the recorder
process's CPU%/RSS and logs them on /pitwall/monitor/* into the same file.

With live:=true it also starts foxglove_bridge so you can watch the same data
live in Foxglove/Lichtblick (ws://<host>:<bridge_port>) while it records.

Notes:
  * Uses the proven `ros2 bag record` recorder (clean finalize on SIGINT from
    launch), not an in-process Python recorder.
  * MCAP is the storage format -> opens directly in Foxglove.
"""

import os
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _truthy(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    output_dir = LaunchConfiguration("output_dir").perform(context)
    topic_regex = LaunchConfiguration("topic_regex").perform(context)
    live = _truthy(LaunchConfiguration("live").perform(context))
    bridge_port = LaunchConfiguration("bridge_port").perform(context)
    if not output_dir:
        output_dir = os.path.join(
            os.path.expanduser("~"), "runs", time.strftime("%Y-%m-%d_%H-%M-%S"))

    record = ExecuteProcess(
        # --include-hidden-topics keeps capture working even if the user sets a
        # hidden ("/_pitwall") prefix via PITWALL_TOPIC_PREFIX.
        cmd=["ros2", "bag", "record", "-s", "mcap", "-o", output_dir,
             "--include-hidden-topics", "--regex", topic_regex],
        output="screen",
        # give rosbag2 time to flush + finalize the MCAP before SIGKILL
        sigterm_timeout="20",
        sigkill_timeout="25",
    )
    monitor = Node(
        package="pitwall",
        executable="monitor_node.py",
        name="pitwall_monitor",
        output="screen",
        parameters=[{"watch_cmdline": "bag record"}],
    )
    actions = [record, monitor]
    if live:
        actions.append(Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": ParameterValue(int(bridge_port), value_type=int),
                "address": "0.0.0.0",
            }],
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "output_dir", default_value="",
            description="MCAP output dir (default: ~/runs/<timestamp>)"),
        DeclareLaunchArgument(
            "topic_regex", default_value="/pitwall/.*",
            description="Regex of topics to capture (add sensors here, e.g. "
                        "'/pitwall/.*|/scan|/imu')"),
        DeclareLaunchArgument(
            "live", default_value="false",
            description="Also start foxglove_bridge for live viewing"),
        DeclareLaunchArgument(
            "bridge_port", default_value="8765",
            description="foxglove_bridge WebSocket port (when live:=true)"),
        OpaqueFunction(function=_launch_setup),
    ])

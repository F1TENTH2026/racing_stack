"""pitwall live view — foxglove_bridge only (no recording).

  ros2 launch pitwall live.launch.py
  ros2 launch pitwall live.launch.py bridge_port:=8765

Starts foxglove_bridge so Foxglove/Lichtblick can connect live over WebSocket
(ws://<host>:<port>) and watch /pitwall/* plus any other ROS topics. Run this on
the (remote) machine that has the sensors/data; connect the viewer from your
laptop. For live + simultaneous recording use record.launch.py with live:=true.

Requires: sudo apt install ros-<distro>-foxglove-bridge
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    port = LaunchConfiguration("bridge_port")
    return LaunchDescription([
        DeclareLaunchArgument(
            "bridge_port", default_value="8765",
            description="foxglove_bridge WebSocket port"),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": ParameterValue(port, value_type=int),
                "address": "0.0.0.0",
            }],
        ),
    ])

"""pitwall.launch.py — open RViz with the pitwall config (Sim Control + telemetry
panel). Standalone entry point (the `unicorn` alias `pitwall` runs this). More
nodes/tools may be added here later.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory('stack_master'), 'config', 'pitwall.rviz')

    rviz_config = LaunchConfiguration('rviz_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config', default_value=default_rviz,
            description='RViz config to load (defaults to stack_master/config/pitwall.rviz)'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='frenet_odom_republisher',
            executable='frenet_odom_republisher_node',
            name='frenet_odom_republisher',
            output='screen',
        ),
    ])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    perception_only = LaunchConfiguration('PerceptionOnly')

    return LaunchDescription([
        DeclareLaunchArgument(
            'PerceptionOnly',
            default_value='false',
            description='Use perception-only service names'),
        Node(
            package='frenet_conversion_server',
            executable='frenet_conversion_server_node',
            name='frenet_conversion_server',
            output='screen',
            parameters=[{'PerceptionOnly': perception_only}],
        ),
    ])

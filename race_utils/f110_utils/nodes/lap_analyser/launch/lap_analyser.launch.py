from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    loc_algo_arg = DeclareLaunchArgument(
        'loc_algo', default_value='slam',
        description='Localization algorithm used (slam / ...)')

    return LaunchDescription([
        loc_algo_arg,
        Node(
            package='lap_analyser',
            executable='lap_analyser',
            name='lap_analyser',
            output='screen',
            parameters=[{'loc_algo': LaunchConfiguration('loc_algo')}],
        ),
    ])

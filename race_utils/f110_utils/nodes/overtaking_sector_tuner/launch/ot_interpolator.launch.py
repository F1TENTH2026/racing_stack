import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('overtaking_sector_tuner'), 'config', 'ot_sectors.yaml')

    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='ot_sectors.yaml with overtaking sector parameters')

    return LaunchDescription([
        params_arg,
        Node(
            package='overtaking_sector_tuner',
            executable='ot_interpolator',
            name='ot_sector_tuner',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])

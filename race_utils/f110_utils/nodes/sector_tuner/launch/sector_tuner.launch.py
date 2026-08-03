import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_save_dir = os.path.join(
        get_package_share_directory('sector_tuner'), 'config')

    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value=default_save_dir,
        description='Directory to write speed_scaling.yaml into')

    return LaunchDescription([
        save_dir_arg,
        Node(
            package='sector_tuner',
            executable='sector_slicer',
            name='sector_slicer',
            output='screen',
            parameters=[{'save_dir': LaunchConfiguration('save_dir')}],
        ),
    ])

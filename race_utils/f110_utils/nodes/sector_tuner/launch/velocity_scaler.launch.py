import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('sector_tuner'), 'config', 'speed_scaling.yaml')

    debug_plot_arg = DeclareLaunchArgument(
        'debug_plot', default_value='False',
        description='Whether to show the debug plot of the scaling')
    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='speed_scaling.yaml with sector parameters')

    return LaunchDescription([
        debug_plot_arg,
        params_arg,
        Node(
            package='sector_tuner',
            executable='velocity_scaler',
            name='velocity_scaler',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'debug_plot': LaunchConfiguration('debug_plot')},
            ],
        ),
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Prefer the stack_master config if present, otherwise fall back to the
    # config shipped inside this package.
    sm_share = get_package_share_directory('state_machine')
    default_config = os.path.join(sm_share, 'config', 'state_machine_params.yaml')

    stack_master_config = ''
    try:
        stack_master_share = get_package_share_directory('stack_master')
        candidate = os.path.join(stack_master_share, 'config', 'state_machine_params.yaml')
        if os.path.exists(candidate):
            stack_master_config = candidate
    except Exception:
        pass

    config = stack_master_config if stack_master_config else default_config

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=config,
        description='Path to the state_machine parameter yaml file',
    )

    return LaunchDescription([
        config_arg,
        Node(
            package='state_machine',
            name='state_machine',
            executable='state_machine',
            output='screen',
            parameters=[LaunchConfiguration('config')],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
    ])

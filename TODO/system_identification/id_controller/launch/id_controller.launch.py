import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('id_controller')
    default_param_file = os.path.join(pkg_share, 'config', 'experiments.yaml')

    experiment_arg = DeclareLaunchArgument(
        'experiment', default_value='5')
    id_param_file_arg = DeclareLaunchArgument(
        'id_param_file', default_value=default_param_file)
    drive_topic_arg = DeclareLaunchArgument(
        'drive_topic',
        default_value='/vesc/high_level/ackermann_cmd_mux/input/nav_1')
    # alternatives: /vesc/high_level/ackermann_cmd_mux/input/ctrl, /drive

    experiment = LaunchConfiguration('experiment')
    id_param_file = LaunchConfiguration('id_param_file')
    drive_topic = LaunchConfiguration('drive_topic')

    id_controller_node = Node(
        package='id_controller',
        executable='controller_node',
        name='id_controller',
        output='screen',
        parameters=[
            id_param_file,
            {'experiment': ParameterValue(experiment, value_type=int)},
        ],
        remappings=[('drive_topic', drive_topic)],
    )

    return LaunchDescription([
        experiment_arg,
        id_param_file_arg,
        drive_topic_arg,
        id_controller_node,
    ])

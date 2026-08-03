# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    ld = LaunchDescription()

    map_yaml_path = LaunchConfiguration('map_yaml_path')
    map_yaml_path_arg = DeclareLaunchArgument(
        'map_yaml_path', description="Path to map YAML file. Passed in via top-level launchfile.")

    ego_odom_topic = LaunchConfiguration('ego_odom_topic')
    ego_odom_topic_arg = DeclareLaunchArgument(
        'ego_odom_topic', default_value='car_state/odom',
        description="Topic name for ego odometry output from gym bridge.")

    publish_tf = LaunchConfiguration('publish_tf')
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf', default_value='true',
        description="Whether gym_bridge publishes map->base_link TF. True only for gt localization mode.")

    ego_scan_out = LaunchConfiguration('ego_scan_out')
    ego_scan_out_arg = DeclareLaunchArgument(
        'ego_scan_out', default_value='/scan_raw',
        description="Topic the ego scan is published on: /scan_raw when virtual_perception "
                    "overlays it onto /scan (default), or /scan to publish straight through "
                    "when virtual injection is off.")

    sim_setup_params = os.path.join(
        get_package_share_directory('stack_master'),
        'config',
        'SIM',
        'sim.yaml')

    config_dict = yaml.safe_load(open(sim_setup_params, 'r'))
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[sim_setup_params,
                    {'map_path': map_yaml_path},
                    {'dynamics': os.path.join(get_package_share_directory('stack_master'), 'config', 'SIM', 'dynamics.yaml')},
                    {'ego_odom_topic': ego_odom_topic},
                    {'publish_tf': publish_tf},
                    # ego-only: the `opponent` package owns the opponent (same as on the real car)
                    {'use_external_opponent': True}],
        # ego scan goes to ego_scan_out (/scan_raw when scan_overlay owns /scan, else /scan)
        remappings=[('/initialpose', '/sim/initialpose'),
                    ('/scan', ego_scan_out)]
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=[
            '-d', os.path.join(get_package_share_directory('stack_master'), 'config', 'pitwall.rviz')],
        remappings=[('/initialpose', '/sim/initialpose')]
    )

    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'config', 'ego_racecar.xacro')])}],
        remappings=[('/robot_description', 'ego_robot_description')]
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'config', 'opp_racecar.xacro')])}],
        remappings=[('/robot_description', 'opp_robot_description')]
    )
    # The opponent (vehicle + controller + scan_augmentor) is provided by the
    # separate `opponent` package, included from low_level.launch.xml so the SAME
    # setup runs on the real car. gym_bridge here is ego-only.
    # TODO: add IMU

    # finalize
    ld.add_action(map_yaml_path_arg)
    ld.add_action(ego_odom_topic_arg)
    ld.add_action(publish_tf_arg)
    ld.add_action(ego_scan_out_arg)
    # Delay RViz so it creates its OpenGL context AFTER the other nodes' startup
    # burst (gym_bridge raycaster build + ~28 nodes) settles. Starting it in that
    # burst intermittently corrupts GL init -> a black viewport (renders, ~30 fps,
    # but shows nothing; data/TF are fine). GPU/OS-agnostic. 1.5s is enough here.
    ld.add_action(TimerAction(period=1.5, actions=[rviz_node]))
    ld.add_action(bridge_node)
    ld.add_action(ego_robot_publisher)
    # Always run the opponent's robot_state_publisher so a runtime-spawned
    # opponent (via the RViz 2D Goal Pose tool) shows up as a car mesh. Its
    # base TF is only published by the bridge while an opponent exists.
    ld.add_action(opp_robot_publisher)

    return ld

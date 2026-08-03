"""KISS-ICP map-based localization (MID360 + IMU).

Usage:
    ros2 launch kiss_icp_localization localization.launch.py [use_rviz:=true]

All configuration is in config/mid360_localization.yaml — there are no map/path
launch arguments. Set the absolute paths there:
  - map_pcd_3d / map_pcd_2d : full vs band-cropped (2.5D) map; active one is
    chosen by `localization_2d` (false=3D, true=2D).
  - ground_yaml             : GLIM ground plane (crop_* geometry); the node reads
    it directly and overrides the config crop_* defaults.
  - track_map_yaml          : 2D detection mask (stage-2 filter + default /map viz).
  - map_2d_yaml             : optional override of the /map viz source.

The only launch arg is use_rviz (spawns RViz with the bundled config).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("kiss_icp_localization")
    config = os.path.join(share, "config", "mid360_localization.yaml")
    return LaunchDescription(
        [
            # ICP's per-iteration parallel region is small; many cores hurt
            # (fork/join overhead). Cap threads unless the user already set it.
            SetEnvironmentVariable(
                "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8")
            ),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("use_lap_manager", default_value="true"),
            Node(
                package="kiss_icp_localization",
                executable="localization_node",
                name="kiss_icp_localization",
                output="screen",
                parameters=[config],
                remappings=[("/kiss_loc/odometry", "/car_state/odom")],
            ),
            # Lap timing + colored trajectory. Reset/started by RViz "2D Pose
            # Estimate" (/initialpose); tracks /car_state/odom. See stack_master.
            Node(
                package="stack_master",
                executable="lap_manager_node.py",
                name="lap_manager",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_lap_manager")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "config", "kiss_loc.rviz")],
                output="log",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )

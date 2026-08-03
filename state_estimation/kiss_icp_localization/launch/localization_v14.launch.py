"""KISS-ICP localization on the local v14 map (convenience launch).

A thin wrapper around the node that hard-points at the local v14 PCD:
    /home/leesh/ifac/kiss_ws/src/kiss_icp_localization/map/v14/map.pcd

This does NOT touch the default config/launch — `localization.launch.py`
still resolves maps via KISS_LOC_MAPS_ROOT exactly as before. This file is
just a local shortcut so you can run:

    ros2 launch kiss_icp_localization localization_v14.launch.py

(the v14 map.pcd is XYZI without normals; the node estimates normals by
PCA at load, so point-to-plane still applies.)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MAP_PCD = "/home/leesh/ifac/kiss_ws/src/kiss_icp_localization/map/v14/map.pcd"


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("kiss_icp_localization")
    config = os.path.join(share, "config", "mid360_localization.yaml")
    nodes = [
        Node(
            package="kiss_icp_localization",
            executable="localization_node",
            name="kiss_icp_localization",
            output="screen",
            parameters=[
                config,
                {
                    "map_pcd_path": MAP_PCD,
                    # BEV detection on. Ground removal + height crop reuse the
                    # GLIM ground plane from ground_lidar.yaml (crop_ground_* /
                    # crop_z_min/max in config) — no separate detect ground params.
                    "detect_en": True,
                },
            ],
        )
    ]
    if LaunchConfiguration("use_rviz").perform(context).lower() == "true":
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "config", "kiss_loc.rviz")],
                output="log",
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable(
                "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8")
            ),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )

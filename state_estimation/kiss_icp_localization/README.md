# kiss_icp_localization

KISS-ICP style map-based LiDAR localization for the **Livox MID360** (with its built-in IMU), built for ROS 2.

Incoming scans are registered against a prior map (PCD) using robust point-to-point / point-to-plane ICP. The MID360's gyro fills the gaps between LiDAR frames (deskew, motion prediction, high-rate odometry). It is a lightweight drop-in alternative to `fast_livo` localization, sharing the same interface (`/livox/lidar`, `/livox/imu`, `/initialpose`).

## Features

- **VoxelHashMap** — static voxel hash of the prior map (point + normal, per-voxel point cap). Fixed 27-neighbor NN search like KISS-ICP. Estimates normals via voxel-neighborhood PCA.
- **Robust ICP** — Gauss-Newton + Geman-McClure; point-to-plane when normals exist, point-to-point otherwise. OpenMP-parallel, with ridge + step clamping against degenerate geometry.
- **Adaptive threshold** — KISS-ICP adaptive correspondence radius driven by model-error RMS.
- **IMU integration** — gyro-only (no accelerometer): scan deskew, inter-scan rotation prediction, IMU-rate propagation. Translation via constant-velocity + EMA.
- **Localization safeguards** (beyond stock KISS-ICP) — divergence gate (reject + coast on prediction, re-anchor after N consecutive rejections), velocity/acceleration caps to prevent runaway estimates in corridor-degenerate sections.

## Dependencies

ROS 2 (`ament_cmake`), with: `rclcpp`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `tf2_ros`, `pcl_conversions`, `livox_ros_driver2`, Eigen.

## Build & Test

```bash
colcon build --packages-select kiss_icp_localization
colcon test  --packages-select kiss_icp_localization   # core gtest (synthetic-room convergence)

# ROS-level smoke test (synthetic map + simulated scan/IMU, ~10s)
python3 test/smoke_test.py

# Real-data validation (bag + map PCD, real-time replay)
python3 test/bag_validation.py [bag_dir] [map_pcd]
```

## Run

```bash
ros2 launch kiss_icp_localization localization.launch.py map:=hall_0609
```

`map:=` is a directory name under the maps root (`<root>/<map>/cloudGlobal.pcd`).
The root resolves in order: `KISS_LOC_MAPS_ROOT` → `FASTLIVO_MAPS_ROOT` → in-repo `stack_master/maps`.

**Topics**

- `/kiss_loc/odometry` — scan-corrected + IMU-rate propagated odometry (single topic)
- `/kiss_loc/scan_aligned` — aligned scan
- `/kiss_loc/map` — prior map (`transient_local`)
- TF: `map → base_link`

The `initial_pose` parameter assumes a start at the map origin. To start elsewhere, re-anchor with RViz `/initialpose`.

## License

MIT — see [LICENSE](LICENSE).

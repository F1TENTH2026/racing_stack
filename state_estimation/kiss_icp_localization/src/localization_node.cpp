// KISS-ICP style map-based localization for Livox MID360 (LiDAR + IMU).
//
// - Prior map (PCD) -> static voxel hash map
// - Incoming scans: IMU-gyro deskew -> voxel downsample -> robust
//   point-to-point ICP against the map (adaptive correspondence threshold)
// - Between LiDAR frames the latest pose is propagated with gyro rotation +
//   constant body velocity and published at IMU rate.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "kiss_icp_localization/adaptive_threshold.hpp"
#include "kiss_icp_localization/bev_detector.hpp"
#include "kiss_icp_localization/registration.hpp"
#include "kiss_icp_localization/se3.hpp"
#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace kiss_loc {

struct ImuSample {
  double t;            // absolute time (s)
  Eigen::Vector3d w;   // bias-removed angular velocity, LiDAR frame (rad/s)
};

struct PendingScan {
  double t_begin = 0.0;  // header stamp (s)
  double t_end = 0.0;    // stamp of last point (s)
  double arrival_wall = 0.0;
  std::vector<Eigen::Vector3d> points;  // sensor frame
  std::vector<float> rel_time;          // per-point time since t_begin (s); empty if unknown
};

class LocalizationNode : public rclcpp::Node {
public:
  LocalizationNode() : Node("kiss_icp_localization") {
    declareParams();
    loadMap();
    initPoseFromParam();

    adaptive_ = std::make_unique<AdaptiveThreshold>(
        initial_threshold_, min_threshold_, min_motion_, adaptive_range_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 50);
    aligned_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/kiss_loc/scan_aligned", 5);
    if (publish_2d_scan_)
      scan_2d_pub_ =
          create_publisher<sensor_msgs::msg::PointCloud2>(scan_2d_topic_, 5);
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/kiss_loc/map", rclcpp::QoS(1).transient_local());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    publishMapCloud();

    if (publish_2d_map_) {
      grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
          map_2d_topic_, rclcpp::QoS(1).transient_local());
      if (load2DMap()) publish2DMap();
    }

    // 2D localization SDF map (scan-to-SDF target): its own yaml so the matching
    // map can differ from the detection track mask. Empty loc_2d_map_yaml falls
    // back to the track map. A missing map is fatal — in 2D mode it IS the map.
    if (localization_2d_) {
      const std::string lp =
          loc_map_path_.empty() ? resolveTrackMapPath() : loc_map_path_;
      if (!loc_mask_.Load(lp)) {
        RCLCPP_FATAL(get_logger(),
                     "2D localization map %s failed to load — set loc_2d_map_yaml "
                     "(or track_map_yaml as fallback)",
                     lp.c_str());
        throw std::runtime_error("2D localization map load failed");
      }
      RCLCPP_INFO(get_logger(), "loaded 2D localization SDF map %s%s", lp.c_str(),
                  loc_map_path_.empty() ? " (fallback: track map)" : "");
    }

    // Detection stage-2 off-track filter uses the track mask. A missing mask is
    // fatal because it's the only spatial filter (silent off-track false pos).
    if (detect_en_ && bev_params_.track_filter) {
      const std::string tp = resolveTrackMapPath();
      if (!track_mask_.Load(tp)) {
        RCLCPP_FATAL(get_logger(),
                     "track mask %s failed to load with detect_track_filter=true "
                     "— fix track_map_yaml or set detect_track_filter:=false",
                     tp.c_str());
        throw std::runtime_error("track mask load failed");
      }
      RCLCPP_INFO(get_logger(), "loaded detection track mask %s (margin %.2f m)",
                  tp.c_str(), bev_params_.track_margin);
    }

    if (detect_en_) {
      const TrackMask *track =
          (bev_params_.track_filter && track_mask_.Valid()) ? &track_mask_ : nullptr;
      detector_ = std::make_unique<BevDetector>(bev_params_, track);
      obstacle_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          obstacle_topic_, 5);
      det_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          detection_topic_, 5);
      obstacle_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
          obstacle_pose_topic_, 5);
      RCLCPP_INFO(get_logger(),
                  "BEV detection enabled: res %.2f m, height band [%.2f, %.2f] "
                  "above GLIM ground n=(%.4f,%.4f,%.4f) off=%.4f, "
                  "track_filter=%d (margin %.2f)",
                  bev_params_.res, bev_params_.z_min, bev_params_.z_max,
                  crop_n_.x(), crop_n_.y(), crop_n_.z(), crop_h_,
                  bev_params_.track_filter, bev_params_.track_margin);
    }

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(200);
    if (input_scan_) {
      scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
          scan_topic_, sensor_qos,
          [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) { onScan(msg); });
    } else {
      pc2_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          lidar_topic_, sensor_qos,
          [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { onPC2(msg); });
    }
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, sensor_qos,
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { onImu(msg); });
    if (use_odom_twist_) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          odom_twist_topic_, rclcpp::SensorDataQoS().keep_last(20),
          [this](const nav_msgs::msg::Odometry::SharedPtr msg) { onOdom(msg); });
    }
    if (use_initial_pose_topic_) {
      initpose_sub_ =
          create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
              "/initialpose", 5,
              [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr
                         msg) { onInitialPose(msg); });
    }

    if (localization_2d_ && input_scan_)
      RCLCPP_INFO(get_logger(),
                  "localization=2D (LaserScan input): no band-crop, scan-to-SDF "
                  "match (3-DoF x,y,yaw) against track mask");
    else if (localization_2d_)
      RCLCPP_INFO(get_logger(),
                  "localization=2D: band-crop scan to height [%.2f, %.2f] m above "
                  "plane n=(%.4f,%.4f,%.4f) off=%.4f, then scan-to-SDF match "
                  "(3-DoF x,y,yaw) against track mask",
                  crop_z_min_, crop_z_max_, crop_n_.x(), crop_n_.y(), crop_n_.z(), crop_h_);
    else
      RCLCPP_INFO(get_logger(), "localization=3D: full scan into ICP (no band crop)");
    RCLCPP_INFO(get_logger(),
                "kiss_icp_localization ready: map %zu pts (voxel %.2f m), "
                "input %s '%s', imu '%s' (imu_en=%d), odom_twist '%s'",
                map_.NumPoints(), map_voxel_size_,
                input_scan_ ? "LaserScan" : "PointCloud2",
                input_scan_ ? scan_topic_.c_str() : lidar_topic_.c_str(),
                imu_topic_.c_str(), imu_en_,
                use_odom_twist_ ? odom_twist_topic_.c_str() : "(none, CV)");
  }

private:
  // ----------------------------- setup -----------------------------
  void declareParams() {
    // separate maps per mode: 3D=full cloud, 2D=band-cropped (2.5D) cloud. The
    // active one (selected below by localization_2d) is loaded for ICP and is the
    // dir base for deriving the 2D raster (map_2d / map_track).
    map_pcd_3d_ = declare_parameter<std::string>("map_pcd_3d", "");
    map_pcd_2d_ = declare_parameter<std::string>("map_pcd_2d", "");
    map_voxel_size_ = declare_parameter<double>("map_voxel_size", 0.5);
    map_max_points_ = declare_parameter<int>("map_max_points_per_voxel", 30);
    // point-to-plane when the map PCD carries normals (fast_livo save_map does)
    use_normals_ = declare_parameter<bool>("use_normals", true);
    scan_voxel_size_ = declare_parameter<double>("scan_voxel_size", 0.35);
    min_range_ = declare_parameter<double>("min_range", 0.3);
    max_range_ = declare_parameter<double>("max_range", 60.0);
    point_filter_num_ = declare_parameter<int>("point_filter_num", 1);

    // ---- 2D / 3D localization + ground-band geometry ----
    // localization_2d=false (3D): no crop, full scan into ICP (range-only).
    // localization_2d=true  (2D) : keep only points whose height above the ground
    //   plane is in [crop_z_min, crop_z_max], so the scan matches a band-cropped
    //   (2.5D) map for a consistent registration. The plane is in the LiDAR frame
    //   — a constant LiDAR<->ground extrinsic from GLIM (normal = R_map_lidar^T *
    //   n_map, offset = n_map.t_lidar + d); height(p) = crop_n_.p + crop_h_.
    //   Applied pre-deskew in keepPoint(). This single bool lives only in the node
    //   config (the old crop_ground_mode also sat in ground_lidar.yaml, which only
    //   carries the plane GEOMETRY below — normal/offset/z_min/z_max).
    localization_2d_ = declare_parameter<bool>("localization_2d", false);
    // input front-end: "3d"=PointCloud2 (default, current behaviour) | "2d"/"scan"
    // = subscribe a sensor_msgs/LaserScan directly. A 2D scan has no 3D structure
    // to ICP, so it FORCES scan-to-SDF matching (localization_2d) and the vertical
    // band-crop / ground-plane geometry are skipped — the points are already on
    // the sensor's 2D plane (z=0). Declared here (before the map/ground-yaml
    // fail-fasts below) so they can branch on it.
    input_mode_ = declare_parameter<std::string>("input_mode", "3d");
    input_scan_ = (input_mode_ == "2d" || input_mode_ == "scan");
    if (input_scan_) localization_2d_ = true;
    // 3D: full-cloud voxel map for ICP — required. 2D: registration is scan-to-SDF
    // against the track mask (see AlignScanToTrackSdf), so the PCD is NOT used for
    // matching; map_pcd_2d is optional and only serves as the /kiss_loc/map viz +
    // the dir anchor for deriving sibling yamls. track_map_yaml is what's required
    // in 2D (enforced when the track mask is loaded below).
    detect_en_ = declare_parameter<bool>("detect_en", false);  // needed by ground-yaml check
    map_pcd_path_ = localization_2d_ ? map_pcd_2d_ : map_pcd_3d_;
    if (map_pcd_path_.empty() && !localization_2d_) {
      RCLCPP_FATAL(get_logger(),
                   "map_pcd_3d is empty — set it in the config (localization_2d=false)");
      throw std::runtime_error("active map path not set");
    }
    auto cn = declare_parameter<std::vector<double>>("crop_ground_normal", {0.0, 0.0, 1.0});
    crop_n_ = (cn.size() == 3) ? Eigen::Vector3d(cn[0], cn[1], cn[2])
                               : Eigen::Vector3d(0.0, 0.0, 1.0);
    if (crop_n_.norm() > 1e-9) crop_n_.normalize();
    crop_h_ = declare_parameter<double>("crop_ground_offset", 0.0);
    crop_z_min_ = declare_parameter<double>("crop_z_min", 0.05);
    crop_z_max_ = declare_parameter<double>("crop_z_max", 0.30);
    // GLIM ground plane: config-only auto-link from the active map folder (or an
    // explicit ground_yaml). Overrides the four crop_* members above if present —
    // replaces the launch param-file layering. Must run before the detection
    // z-band (bp.z_min/z_max) is copied from crop_z_min_/max_ below.
    ground_yaml_ = declare_parameter<std::string>("ground_yaml", "");
    loadGroundYaml();

    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/livox/lidar");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/livox/imu");
    // 2D-input (input_mode=2d) LaserScan topic.
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    // Wheel-odometry topic whose twist drives the TRANSLATION prediction (gyro
    // drives rotation). Empty -> translation falls back to the ICP-derived
    // constant-velocity estimate. Set "/vesc/odom" on the unicorn car.
    odom_twist_topic_ = declare_parameter<std::string>("odom_twist_topic", "");
    use_odom_twist_ = !odom_twist_topic_.empty();
    // livox driver with use_system_timestamp stamps the header with now() at
    // publish time, i.e. at the END of the 100 ms accumulation window
    stamp_at_scan_end_ = declare_parameter<bool>("stamp_at_scan_end", true);

    imu_en_ = declare_parameter<bool>("imu_en", true);
    deskew_en_ = declare_parameter<bool>("deskew_en", true);
    imu_rate_odom_ = declare_parameter<bool>("imu_rate_odom", true);
    imu_init_samples_ = declare_parameter<int>("imu_init_samples", 100);
    auto r_il = declare_parameter<std::vector<double>>(
        "extrinsic_R_il", {1, 0, 0, 0, 1, 0, 0, 0, 1});
    if (r_il.size() == 9) {
      R_il_ = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r_il.data());
    } else {
      R_il_.setIdentity();
    }

    max_iterations_ = declare_parameter<int>("max_iterations", 50);
    convergence_eps_ = declare_parameter<double>("convergence_eps", 2e-3);
    initial_threshold_ = declare_parameter<double>("initial_threshold", 1.0);
    min_threshold_ = declare_parameter<double>("min_threshold", 0.1);
    min_motion_ = declare_parameter<double>("min_motion", 0.05);
    // characteristic range for the rotation term of the adaptive threshold.
    // Tightening this (e.g. to the true indoor scene scale ~15 m) measurably
    // LOSES map-lock during aggressive driving — the inflated threshold is
    // robustness margin, keep it at sensor max range like upstream KISS-ICP
    adaptive_range_ = declare_parameter<double>("adaptive_range", 60.0);
    vel_smoothing_ = declare_parameter<double>("vel_smoothing", 0.3);
    // 2D scan-to-SDF complementary-filter gain: fraction of the SDF fix applied
    // per scan (1=full/raw, lower=smoother, gyro carries fast motion). Cuts the
    // per-scan yaw/xy jitter inherent to a few-point discretized-SDF fit.
    corr_gain_ = declare_parameter<double>("loc_2d_corr_gain", 1.0);
    reject_trans_ = declare_parameter<double>("reject_trans", 2.0);
    reject_rot_deg_ = declare_parameter<double>("reject_rot_deg", 30.0);
    reject_recover_count_ = declare_parameter<int>("reject_recover_count", 3);
    max_velocity_ = declare_parameter<double>("max_velocity", 15.0);
    max_accel_ = declare_parameter<double>("max_accel", 10.0);

    initial_pose_ = declare_parameter<std::vector<double>>(
        "initial_pose", {0, 0, 0, 0, 0, 0});  // x y z roll pitch yaw
    use_initial_pose_topic_ = declare_parameter<bool>("use_initial_pose_topic", true);

    print_stats_ = declare_parameter<bool>("print_stats", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/kiss_loc/odometry");
    // 2D occupancy grid: published by the node itself (no nav2_map_server).
    // map_2d_yaml empty -> derive <map_pcd dir>/map_2d.yaml.
    publish_2d_map_ = declare_parameter<bool>("publish_2d_map", true);
    map_2d_topic_ = declare_parameter<std::string>("map_2d_topic", "/map");
    map_2d_yaml_ = declare_parameter<std::string>("map_2d_yaml", "");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    publish_aligned_scan_ = declare_parameter<bool>("publish_aligned_scan", true);
    // 2D-ized aligned scan: the ground-band slab (z kept) of the aligned scan in
    // the map frame, for 2D consumers (costmap / scan matchers). Independent of
    // localization_2d — the band is computed here regardless of the ICP input.
    publish_2d_scan_ = declare_parameter<bool>("publish_2d_scan", true);
    scan_2d_topic_ = declare_parameter<std::string>("scan_2d_topic", "/kiss_loc/scan_2d");

    // ---- BEV object detection (unmapped statics + opponents) ----
    // detect_en_ already declared above (ground-yaml fail-fast needs it).
    BevParams bp;
    bp.res = declare_parameter<double>("detect_res", 0.2);
    // ground removal + height crop reuse the GLIM ground plane and z-band from
    // ground_lidar.yaml (crop_ground_normal/offset, crop_z_min/z_max) — single
    // source of truth with the localization input crop. The plane is in the
    // sensor frame; runDetection() rotates it into the map frame per pose.
    bp.z_min = crop_z_min_;
    bp.z_max = crop_z_max_;
    bp.eps = declare_parameter<double>("detect_eps", 0.2);
    bp.min_samples = declare_parameter<int>("detect_min_samples", 4);
    bp.min_cluster_cells = declare_parameter<int>("detect_min_cluster_cells", 2);
    bp.track_gate = declare_parameter<double>("detect_track_gate", 1.0);
    bp.moving_speed = declare_parameter<double>("detect_moving_speed", 0.5);
    bp.max_misses = declare_parameter<int>("detect_max_misses", 5);
    // stage-2 subtraction: reject detections outside the 2D track mask
    // (GLIM map_track). Empty track_map_yaml -> <map dir>/map_track.yaml.
    bp.track_filter = declare_parameter<bool>("detect_track_filter", false);
    bp.track_margin = declare_parameter<double>("detect_track_margin", 0.3);
    track_map_path_ = declare_parameter<std::string>("track_map_yaml", "");
    // dedicated 2D-localization SDF map (yaml+pgm); empty -> fall back to track map
    loc_map_path_ = declare_parameter<std::string>("loc_2d_map_yaml", "");
    bev_params_ = bp;
    arrow_scale_ = declare_parameter<double>("detect_arrow_scale", 0.5);
    // output topics (yaml-configurable): foreground/obstacle cloud + detection
    // markers. Defaults preserve the historical /kiss_loc/* names.
    obstacle_topic_ =
        declare_parameter<std::string>("detect_obstacle_topic", "/kiss_loc/obstacles");
    detection_topic_ =
        declare_parameter<std::string>("detect_marker_topic", "/kiss_loc/detections");
    // DBSCAN cluster centers as PoseArray (map frame) — e.g. feed MPCC's
    // /external_obstacles. Each pose.position = cluster bbox center.
    obstacle_pose_topic_ =
        declare_parameter<std::string>("detect_pose_topic", "/kiss_loc/obstacle_poses");
  }

  // detection mask (= 2D track raster) yaml: explicit track_map_yaml, else
  // <active map dir>/map_track.yaml. Used both for the BEV stage-2 filter and,
  // by default, as the /map visualization backdrop (see load2DMap).
  std::string resolveTrackMapPath() const {
    if (!track_map_path_.empty()) return track_map_path_;
    if (map_pcd_path_.empty()) return "";
    const auto slash = map_pcd_path_.find_last_of('/');
    const std::string dir =
        (slash == std::string::npos) ? "" : map_pcd_path_.substr(0, slash + 1);
    return dir + "map_track.yaml";
  }

  // GLIM ground plane auto-load (config-only; no launch param-file layering).
  // Reads ground_yaml_ (or <active map dir>/ground_lidar.yaml) and overrides the
  // crop_* geometry. It's a ROS param file but we only need four keys, so scan
  // lines rather than pull in a yaml parser. crop_ground_mode (legacy) ignored.
  void loadGroundYaml() {
    std::string path = ground_yaml_;
    if (path.empty()) {
      if (map_pcd_path_.empty()) return;
      const auto slash = map_pcd_path_.find_last_of('/');
      const std::string dir =
          (slash == std::string::npos) ? "" : map_pcd_path_.substr(0, slash + 1);
      path = dir + "ground_lidar.yaml";
    }
    std::ifstream f(path);
    if (!f) {
      // The band geometry is wrong if the ground plane is missing (config default
      // is the trivial identity plane). Fail fast when the path was set explicitly,
      // or when 2D localization / detection actually needs the band. Only a plain
      // 3D run with no detection may proceed on the config crop_* defaults.
      const bool explicit_path = !ground_yaml_.empty();
      // scan-input 2D needs no band geometry (points are already 2D); only a
      // PointCloud2-fed 2D crop or detection requires the ground plane.
      const bool need_band = (localization_2d_ && !input_scan_) || detect_en_;
      if (explicit_path || need_band) {
        RCLCPP_FATAL(get_logger(),
                     "ground plane yaml not readable: %s (explicit=%d "
                     "localization_2d=%d detect_en=%d) — the z-band crop needs it",
                     path.c_str(), explicit_path, localization_2d_, detect_en_);
        throw std::runtime_error("ground_lidar.yaml not loaded");
      }
      RCLCPP_INFO(get_logger(),
                  "no ground_lidar.yaml at %s -> 3D run, config crop params",
                  path.c_str());
      return;
    }
    std::string line;
    while (std::getline(f, line)) {
      const auto hash = line.find('#');
      if (hash != std::string::npos) line = line.substr(0, hash);
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string key = line.substr(0, colon), val = line.substr(colon + 1);
      key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
      if (key == "crop_ground_normal") {
        for (char &c : val)
          if (c == '[' || c == ']' || c == ',') c = ' ';
        std::istringstream iss(val);
        double a, b, c;
        if (iss >> a >> b >> c) {
          crop_n_ = Eigen::Vector3d(a, b, c);
          if (crop_n_.norm() > 1e-9) crop_n_.normalize();
        }
      } else if (key == "crop_ground_offset") {
        crop_h_ = std::atof(val.c_str());
      } else if (key == "crop_z_min") {
        crop_z_min_ = std::atof(val.c_str());
      } else if (key == "crop_z_max") {
        crop_z_max_ = std::atof(val.c_str());
      }
    }
    RCLCPP_INFO(get_logger(), "ground crop auto-linked from %s", path.c_str());
  }

  void loadMap() {
    if (map_pcd_path_.empty()) {
      // only reachable in 2D mode (3D fails earlier) — scan-to-SDF needs no PCD;
      // skip the voxel map and the /kiss_loc/map cloud viz.
      RCLCPP_INFO(get_logger(),
                  "no map PCD (localization_2d, scan-to-SDF) — voxel map skipped");
      return;
    }
    pcl::PointCloud<pcl::PointNormal> cloud;  // missing normal fields load as 0
    if (pcl::io::loadPCDFile<pcl::PointNormal>(map_pcd_path_, cloud) < 0) {
      RCLCPP_FATAL(get_logger(), "failed to load map PCD: %s", map_pcd_path_.c_str());
      throw std::runtime_error("failed to load map PCD");
    }
    std::vector<Eigen::Vector3d> pts, normals;
    pts.reserve(cloud.size());
    normals.reserve(cloud.size());
    size_t n_valid_normals = 0;
    for (const auto &p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      pts.emplace_back(p.x, p.y, p.z);
      normals.emplace_back(p.normal_x, p.normal_y, p.normal_z);
      if (normals.back().allFinite() && normals.back().norm() > 0.5)
        ++n_valid_normals;
    }
    // 2D mode loads the PCD only for the /kiss_loc/map backdrop — it is not the
    // ICP target (scan-to-SDF is), so build point-only and skip normal PCA.
    const bool want_normals = use_normals_ && !localization_2d_;
    const bool normals_ok =
        want_normals && n_valid_normals > pts.size() / 2;
    map_ = VoxelHashMap(map_voxel_size_, map_max_points_);
    map_.Build(pts, normals_ok ? normals : std::vector<Eigen::Vector3d>{});
    if (want_normals && !map_.HasNormals()) {
      // fast_livo save_map writes zero normals — estimate per-voxel by PCA
      const auto t0 = std::chrono::steady_clock::now();
      map_.EstimateNormals();
      RCLCPP_INFO(get_logger(), "estimated map normals by voxel PCA (%.1f s)",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count());
    }
    map_cloud_raw_ = std::move(pts);
    RCLCPP_INFO(get_logger(),
                "loaded map %s: %zu raw pts -> %zu in voxel map (%s)",
                map_pcd_path_.c_str(), map_cloud_raw_.size(), map_.NumPoints(),
                map_.HasNormals() ? "point-to-plane" : "point-to-point");
  }

  void initPoseFromParam() {
    T_ = Eigen::Isometry3d::Identity();
    if (initial_pose_.size() == 6) {
      T_.translation() =
          Eigen::Vector3d(initial_pose_[0], initial_pose_[1], initial_pose_[2]);
      T_.linear() =
          (Eigen::AngleAxisd(initial_pose_[5], Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(initial_pose_[4], Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(initial_pose_[3], Eigen::Vector3d::UnitX()))
              .toRotationMatrix();
    }
    // 2D mode pins the pose to this z and fixes roll/pitch to the LiDAR mount
    // tilt: R_level_ levels the ground normal (crop_n_, from ground_lidar.yaml)
    // onto the map vertical, so the planar attitude is Rz(yaw) * R_level_.
    plane_z_ = T_.translation().z();
    R_level_ = Eigen::Quaterniond::FromTwoVectors(crop_n_, Eigen::Vector3d::UnitZ())
                   .toRotationMatrix();
    if (localization_2d_) T_ = planarize(T_);
    T_prop_ = T_;
  }

  void publishMapCloud() {
    // Downsample for visualization only
    auto ds = VoxelDownsample(map_cloud_raw_, 0.2);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(ds.size());
    for (const auto &p : ds) cloud.emplace_back(p.x(), p.y(), p.z());
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = now();
    map_pub_->publish(msg);
  }

  // Parse a standard map_server YAML (image/resolution/origin/negate/
  // occupied_thresh/free_thresh) + binary P5 PGM into an OccupancyGrid.
  // Self-contained (no nav2_map_server / yaml-cpp). Returns false on any
  // missing/unreadable file so the node just skips /map publishing.
  bool load2DMap() {
    // viz backdrop source: explicit map_2d_yaml override, else the detection
    // mask (track map) just configured, else <active map dir>/map_2d.yaml.
    std::string yaml_path = map_2d_yaml_;
    if (yaml_path.empty()) yaml_path = resolveTrackMapPath();
    if (yaml_path.empty()) {
      if (map_pcd_path_.empty()) return false;
      const auto slash = map_pcd_path_.find_last_of('/');
      const std::string dir =
          (slash == std::string::npos) ? "." : map_pcd_path_.substr(0, slash);
      yaml_path = dir + "/map_2d.yaml";
    }
    std::ifstream yf(yaml_path);
    if (!yf) {
      RCLCPP_WARN(get_logger(), "2D map yaml not found: %s -> skip /map publish",
                  yaml_path.c_str());
      return false;
    }
    auto trim = [](std::string &s) {
      const size_t a = s.find_first_not_of(" \t\r\n");
      if (a == std::string::npos) { s.clear(); return; }
      const size_t b = s.find_last_not_of(" \t\r\n");
      s = s.substr(a, b - a + 1);
    };
    std::string image;
    double resolution = 0.05, ox = 0, oy = 0, oyaw = 0;
    int negate = 0;
    double occ_th = 0.65, free_th = 0.196;
    std::string line;
    while (std::getline(yf, line)) {
      const auto hash = line.find('#');
      if (hash != std::string::npos) line = line.substr(0, hash);
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string key = line.substr(0, colon), val = line.substr(colon + 1);
      trim(key); trim(val);
      if (key == "image") image = val;
      else if (key == "resolution") resolution = std::stod(val);
      else if (key == "negate") negate = std::stoi(val);
      else if (key == "occupied_thresh") occ_th = std::stod(val);
      else if (key == "free_thresh") free_th = std::stod(val);
      else if (key == "origin") {
        for (char &c : val) if (c == '[' || c == ']' || c == ',') c = ' ';
        std::istringstream iss(val);
        iss >> ox >> oy >> oyaw;
      }
    }
    if (image.empty()) return false;
    std::string img_path = image;
    if (image[0] != '/') {
      const auto slash = yaml_path.find_last_of('/');
      const std::string dir =
          (slash == std::string::npos) ? "." : yaml_path.substr(0, slash);
      img_path = dir + "/" + image;
    }
    std::ifstream img(img_path, std::ios::binary);
    if (!img) {
      RCLCPP_WARN(get_logger(), "2D map image not found: %s", img_path.c_str());
      return false;
    }
    std::string magic;
    img >> magic;
    if (magic != "P5") {
      RCLCPP_WARN(get_logger(), "2D map PGM not binary P5 (%s): %s",
                  magic.c_str(), img_path.c_str());
      return false;
    }
    // read width/height/maxval, skipping '#' comment lines
    auto read_uint = [&](std::istream &is) -> long {
      while (true) {
        const int c = is.peek();
        if (c == EOF) return -1;
        if (c == '#') { std::string d; std::getline(is, d); continue; }
        if (std::isspace(c)) { is.get(); continue; }
        break;
      }
      long v = -1;
      is >> v;
      return v;
    };
    const long w = read_uint(img), h = read_uint(img), maxval = read_uint(img);
    if (w <= 0 || h <= 0 || maxval <= 0) return false;
    img.get();  // consume the single whitespace after maxval
    std::vector<uint8_t> pix(static_cast<size_t>(w) * h);
    img.read(reinterpret_cast<char *>(pix.data()),
             static_cast<std::streamsize>(pix.size()));
    if (static_cast<size_t>(img.gcount()) != pix.size()) {
      RCLCPP_WARN(get_logger(), "2D map PGM truncated: %s", img_path.c_str());
      return false;
    }
    grid_msg_ = nav_msgs::msg::OccupancyGrid();
    grid_msg_.header.frame_id = map_frame_;
    grid_msg_.info.resolution = static_cast<float>(resolution);
    grid_msg_.info.width = static_cast<uint32_t>(w);
    grid_msg_.info.height = static_cast<uint32_t>(h);
    grid_msg_.info.origin.position.x = ox;
    grid_msg_.info.origin.position.y = oy;
    grid_msg_.info.origin.orientation.z = std::sin(oyaw * 0.5);
    grid_msg_.info.origin.orientation.w = std::cos(oyaw * 0.5);
    grid_msg_.data.resize(static_cast<size_t>(w) * h);
    // PGM row 0 is the top; OccupancyGrid row 0 is the bottom -> flip vertically.
    for (long y = 0; y < h; ++y) {
      for (long x = 0; x < w; ++x) {
        const uint8_t v = pix[static_cast<size_t>(y) * w + x];
        const double p = (negate ? v : 255 - v) / 255.0;  // occupancy prob
        int8_t occ = -1;
        if (p > occ_th) occ = 100;
        else if (p < free_th) occ = 0;
        const long gy = h - 1 - y;
        grid_msg_.data[static_cast<size_t>(gy) * w + x] = occ;
      }
    }
    grid_ready_ = true;
    RCLCPP_INFO(get_logger(),
                "2D map loaded: %ldx%ld res %.3f origin (%.2f,%.2f) <- %s",
                w, h, resolution, ox, oy, img_path.c_str());
    return true;
  }

  void publish2DMap() {
    if (!grid_ready_ || !grid_pub_) return;
    grid_msg_.header.stamp = now();
    grid_msg_.info.map_load_time = now();
    grid_pub_->publish(grid_msg_);
  }

  // --------------------------- callbacks ---------------------------
  // scan.t_begin holds the header stamp on entry; place the scan window
  // around it according to where the driver anchors the stamp
  void finalizeScanTimes(PendingScan &scan, float max_rel) const {
    const double header_t = scan.t_begin;
    if (stamp_at_scan_end_ && max_rel > 0.0f) {
      scan.t_begin = header_t - max_rel;
      scan.t_end = header_t;
    } else {
      scan.t_end = header_t + max_rel;
    }
  }

  void onPC2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    PendingScan scan;
    scan.t_begin = rclcpp::Time(msg->header.stamp).seconds();

    // optional per-point time field
    enum class TimeField { kNone, kOffsetU32Ns, kTimeF32, kTimestampF64 };
    TimeField tfield = TimeField::kNone;
    for (const auto &f : msg->fields) {
      if (f.name == "offset_time" &&
          f.datatype == sensor_msgs::msg::PointField::UINT32)
        tfield = TimeField::kOffsetU32Ns;
      else if (f.name == "time" && f.datatype == sensor_msgs::msg::PointField::FLOAT32)
        tfield = TimeField::kTimeF32;
      else if (f.name == "timestamp" &&
               f.datatype == sensor_msgs::msg::PointField::FLOAT64)
        tfield = TimeField::kTimestampF64;
    }

    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y"),
        iz(*msg, "z");
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<uint32_t>> it_off;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_time;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<double>> it_ts;
    if (tfield == TimeField::kOffsetU32Ns)
      it_off = std::make_unique<sensor_msgs::PointCloud2ConstIterator<uint32_t>>(
          *msg, "offset_time");
    else if (tfield == TimeField::kTimeF32)
      it_time =
          std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "time");
    else if (tfield == TimeField::kTimestampF64)
      it_ts = std::make_unique<sensor_msgs::PointCloud2ConstIterator<double>>(
          *msg, "timestamp");

    float max_rel = 0.0f;
    size_t i = 0;
    double ts0 = std::numeric_limits<double>::quiet_NaN();
    for (; ix != ix.end(); ++ix, ++iy, ++iz, ++i) {
      float rt = 0.0f;
      if (it_off) {
        rt = static_cast<float>(**it_off) * 1e-9f;
        ++(*it_off);
      } else if (it_time) {
        rt = **it_time;
        ++(*it_time);
      } else if (it_ts) {
        // livox driver: absolute device time (ns) — may differ from the
        // header stamp (system time), so take it relative to the first point
        double v = **it_ts;
        ++(*it_ts);
        if (v > 1e12) v *= 1e-9;  // ns -> s
        if (std::isnan(ts0)) ts0 = v;
        rt = static_cast<float>(v - ts0);
      }
      if (point_filter_num_ > 1 && static_cast<int>(i % point_filter_num_) != 0)
        continue;
      if (!keepPoint(*ix, *iy, *iz)) continue;
      scan.points.emplace_back(*ix, *iy, *iz);
      if (tfield != TimeField::kNone) {
        scan.rel_time.push_back(rt);
        max_rel = std::max(max_rel, rt);
      }
    }
    finalizeScanTimes(scan, max_rel);
    enqueueScan(std::move(scan));
  }

  // 2D-input front-end: a planar LaserScan -> (x, y, 0) points in the sensor
  // frame. No vertical band-crop (already 2D); range-gated only. time_increment
  // (if the driver sets it) gives per-point times so gyro deskew still applies.
  void onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    PendingScan scan;
    scan.t_begin = rclcpp::Time(msg->header.stamp).seconds();
    const size_t n = msg->ranges.size();
    scan.points.reserve(n);
    const bool has_time = msg->time_increment > 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float r = msg->ranges[i];
      if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;
      if (r < min_range_ || r > max_range_) continue;
      const float a = msg->angle_min + static_cast<float>(i) * msg->angle_increment;
      scan.points.emplace_back(r * std::cos(a), r * std::sin(a), 0.0);
      if (has_time) {
        const float rt = static_cast<float>(i) * msg->time_increment;
        scan.rel_time.push_back(rt);
        max_rel = std::max(max_rel, rt);
      }
    }
    finalizeScanTimes(scan, max_rel);
    enqueueScan(std::move(scan));
  }

  // Wheel odometry: its body twist drives the TRANSLATION prediction (gyro keeps
  // rotation). Overwrites v_body_ directly; the ICP-derived velocity update in
  // processScan is skipped when use_odom_twist_ is set. Single-threaded executor
  // serialises this with processScan/onImu, so the shared v_body_ needs no lock.
  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    v_body_.x() = msg->twist.twist.linear.x;
    v_body_.y() = msg->twist.twist.linear.y;  // ~0 on a car
    v_body_.z() = 0.0;
  }

  bool keepPoint(float x, float y, float z) const {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const double r2 = double(x) * x + double(y) * y + double(z) * z;
    if (r2 <= min_range_ * min_range_ || r2 >= max_range_ * max_range_) return false;
    if (localization_2d_) {
      const double hgt = crop_n_.x() * x + crop_n_.y() * y + crop_n_.z() * z + crop_h_;
      if (hgt < crop_z_min_ || hgt > crop_z_max_) return false;
    }
    return true;
  }

  void onImu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    const Eigen::Vector3d w_raw(msg->angular_velocity.x, msg->angular_velocity.y,
                                msg->angular_velocity.z);

    if (!bias_ready_) {
      bias_acc_.push_back(w_raw);
      if (static_cast<int>(bias_acc_.size()) >= imu_init_samples_) {
        gyro_bias_.setZero();
        for (const auto &w : bias_acc_) gyro_bias_ += w;
        gyro_bias_ /= bias_acc_.size();
        bias_acc_.clear();
        bias_ready_ = true;
        RCLCPP_INFO(get_logger(), "gyro bias initialized: [%.5f %.5f %.5f] rad/s",
                    gyro_bias_.x(), gyro_bias_.y(), gyro_bias_.z());
      }
      return;
    }

    // angular velocity in LiDAR frame
    const Eigen::Vector3d w = R_il_.transpose() * (w_raw - gyro_bias_);
    if (!imu_buf_.empty() && t <= imu_buf_.back().t) return;  // out-of-order
    imu_buf_.push_back({t, w});
    while (!imu_buf_.empty() && imu_buf_.front().t < t - 10.0) imu_buf_.pop_front();

    // high-rate propagated odometry between LiDAR frames
    if (imu_rate_odom_ && have_first_fix_) {
      const double dt = t - t_prop_;
      if (t_prop_ > 0.0 && dt > 0.0 && dt < 0.1) {
        T_prop_.linear() = T_prop_.rotation() * So3Exp(w * dt);
        T_prop_.translation() += T_prop_.rotation() * (v_body_ * dt);
        publishOdom(T_prop_, t);
      }
      t_prop_ = t;
    }

    processPending();
  }

  void onInitialPose(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    const auto &p = msg->pose.pose.position;
    const auto &q = msg->pose.pose.orientation;
    T.translation() = Eigen::Vector3d(p.x, p.y, p.z);
    T.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
    // RViz 2D Pose Estimate has z = 0; keep current z to stay on the map floor
    if (have_first_fix_) T.translation().z() = T_.translation().z();
    if (localization_2d_) {
      plane_z_ = T.translation().z();  // re-anchor the plane height
      T = planarize(T);
    }
    T_ = T;
    T_prop_ = T;
    v_body_.setZero();
    adaptive_->Reset();
    RCLCPP_WARN(get_logger(), "re-anchored from /initialpose: [%.2f %.2f %.2f]",
                T.translation().x(), T.translation().y(), T.translation().z());
    // re-publish both map layers so RViz (and any late subscriber) refreshes
    publishMapCloud();
    publish2DMap();
  }

  // ------------------------- scan processing -------------------------
  void enqueueScan(PendingScan &&scan) {
    if (scan.points.empty()) return;
    scan.arrival_wall = nowSec();
    pending_.push_back(std::move(scan));
    while (pending_.size() > 20) {
      pending_.pop_front();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "scan queue overflow, dropping oldest");
    }
    processPending();
  }

  double nowSec() { return get_clock()->now().seconds(); }

  void processPending() {
    if (imu_en_ && !bias_ready_) {
      // gyro bias still initializing — scans from this period are useless
      // (no deskew, no prediction) and the platform should be static anyway
      if (pending_.size() > 1) pending_.erase(pending_.begin(), pending_.end() - 1);
      return;
    }
    while (!pending_.empty()) {
      const auto &front = pending_.front();
      bool imu_ok = !imu_en_ || (bias_ready_ && !imu_buf_.empty() &&
                                 imu_buf_.back().t >= front.t_end);
      if (!imu_ok) {
        // wait for IMU coverage, but never stall on a dead IMU stream
        if (nowSec() - front.arrival_wall < 0.3) return;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "processing scan without full IMU coverage");
      }
      PendingScan scan = std::move(pending_.front());
      pending_.pop_front();
      processScan(scan);
    }
  }

  // Integrate gyro over [ta, tb]; returns rotation of frame(tb) w.r.t. frame(ta).
  Eigen::Matrix3d integrateGyro(double ta, double tb) const {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (imu_buf_.empty() || tb <= ta) return R;
    double t_cur = ta;
    for (size_t i = 0; i < imu_buf_.size(); ++i) {
      const auto &s = imu_buf_[i];
      if (s.t <= t_cur) continue;
      const double t_next = std::min(s.t, tb);
      R = R * So3Exp(s.w * (t_next - t_cur));
      t_cur = t_next;
      if (t_cur >= tb) break;
    }
    if (t_cur < tb && !imu_buf_.empty())
      R = R * So3Exp(imu_buf_.back().w * (tb - t_cur));
    return R;
  }

  void processScan(const PendingScan &scan) {
    const auto t_start = std::chrono::steady_clock::now();
    auto ms_since = [](const std::chrono::steady_clock::time_point &t0) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
          .count();
    };

    // 1) deskew to scan end using gyro rotation + constant body velocity
    std::vector<Eigen::Vector3d> pts;
    const bool do_deskew = imu_en_ && deskew_en_ && bias_ready_ &&
                           !scan.rel_time.empty() && scan.t_end > scan.t_begin;
    if (do_deskew) {
      // Rotation timeline relative to scan begin, sampled at IMU timestamps.
      // ws[k] is the angular velocity active on segment (ts[k-1], ts[k]].
      std::vector<double> ts{scan.t_begin};
      std::vector<Eigen::Matrix3d> Rs{Eigen::Matrix3d::Identity()};
      std::vector<Eigen::Vector3d> ws{Eigen::Vector3d::Zero()};
      for (const auto &s : imu_buf_) {
        if (s.t <= scan.t_begin || s.t > scan.t_end + 0.01) continue;
        Rs.push_back(Rs.back() * So3Exp(s.w * (s.t - ts.back())));
        ts.push_back(s.t);
        ws.push_back(s.w);
      }
      const Eigen::Vector3d w_last = imu_buf_.empty()
                                         ? Eigen::Vector3d::Zero()
                                         : imu_buf_.back().w;
      auto rotAt = [&](double t) -> Eigen::Matrix3d {
        auto it = std::upper_bound(ts.begin(), ts.end(), t);
        const size_t idx = (it == ts.begin()) ? 0 : (it - ts.begin() - 1);
        const Eigen::Vector3d w = (idx + 1 < ts.size()) ? ws[idx + 1] : w_last;
        return Rs[idx] * So3Exp(w * std::max(0.0, t - ts[idx]));
      };
      const Eigen::Matrix3d R_end = rotAt(scan.t_end);
      pts.reserve(scan.points.size());
      for (size_t i = 0; i < scan.points.size(); ++i) {
        const double ti = scan.t_begin + scan.rel_time[i];
        const Eigen::Matrix3d R_rel = R_end.transpose() * rotAt(ti);
        pts.push_back(R_rel * scan.points[i] + v_body_ * (ti - scan.t_end));
      }
    } else {
      pts = scan.points;
    }

    // 2) downsample
    const auto ds = VoxelDownsample(pts, scan_voxel_size_);
    if (ds.size() < 20) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "too few points after downsampling (%zu), skipping",
                           ds.size());
      return;
    }

    // 3) motion prediction
    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    const double dt = (last_scan_end_ > 0.0) ? scan.t_end - last_scan_end_ : 0.0;
    if (dt > 0.0 && dt < 0.5) {
      if (imu_en_ && bias_ready_)
        delta.linear() = integrateGyro(last_scan_end_, scan.t_end);
      delta.translation() = v_body_ * dt;
    }
    const Eigen::Isometry3d T_pred = T_ * delta;

    // 4) registration
    const double prep_ms = ms_since(t_start);
    const auto t_icp = std::chrono::steady_clock::now();
    const double th = adaptive_->ComputeThreshold();
    // 2D: flatten the band slab onto the track-mask SDF (3-DoF x,y,yaw);
    // 3D: full robust point-to-plane ICP against the voxel map.
    auto result =
        localization_2d_
            ? AlignScanToTrackSdf(ds, loc_mask_, T_pred, th, th / 3.0,
                                  max_iterations_, convergence_eps_)
            : AlignScanToMap(ds, map_, T_pred, th, th / 3.0, max_iterations_,
                             convergence_eps_, use_normals_);
    // 2D: SDF observes only (x,y,yaw) — pin z, drop roll/pitch so the unobserved
    // axes can't dead-reckon the pose off the ground plane (and so the next
    // prediction's forward velocity stays in-plane instead of leaking into z).
    if (localization_2d_) result.pose = planarize(result.pose);
    const double icp_ms = ms_since(t_icp);

    // 5) divergence gate — an isolated fix jumping away from the prediction
    // is treated as an ICP glitch and coasted over; if it persists, the
    // prediction is what's wrong, so re-anchor to the registration result
    const Eigen::Isometry3d dev = T_pred.inverse() * result.pose;
    const double dev_t = dev.translation().norm();
    const double dev_r = RotationAngle(dev.rotation());
    bool reanchored = false;
    if (dev_t > reject_trans_ || dev_r > reject_rot_deg_ * M_PI / 180.0) {
      ++consecutive_rejects_;
      if (consecutive_rejects_ < reject_recover_count_) {
        RCLCPP_WARN(get_logger(),
                    "registration rejected (dev %.2f m / %.1f deg, corr %d) — "
                    "coasting on prediction (%d consecutive)",
                    dev_t, dev_r * 180.0 / M_PI, result.num_correspondences,
                    consecutive_rejects_);
        T_ = localization_2d_ ? planarize(T_pred) : T_pred;
        last_scan_end_ = scan.t_end;
        T_prop_ = T_;
        t_prop_ = scan.t_end;
        publishOdom(T_, scan.t_end);
        return;
      }
      RCLCPP_WARN(get_logger(),
                  "re-anchoring to registration result after %d rejects "
                  "(dev %.2f m / %.1f deg)",
                  consecutive_rejects_, dev_t, dev_r * 180.0 / M_PI);
      adaptive_->Reset();
      v_body_.setZero();
      reanchored = true;
    }
    consecutive_rejects_ = 0;
    if (!reanchored) adaptive_->UpdateModelDeviation(dev);

    // 2D complementary low-pass: blend the (noisy) SDF fix toward the smooth
    // gyro/CV prediction to suppress per-scan yaw/xy chatter. Done after the
    // divergence gate (which must see the raw fix) and skipped on re-anchor
    // (there we want the full jump). Velocity below is then derived from the
    // smoothed pose, so the high-rate propagation between scans is smooth too.
    if (localization_2d_ && corr_gain_ < 1.0 && !reanchored)
      result.pose = blendPlanar(T_pred, result.pose, corr_gain_);

    // 6) state update — velocity from consecutive fixes, with physical
    // accel/speed limits: in corridor-degenerate stretches ICP can't observe
    // the along-track direction, and an unbounded velocity estimate feeds
    // back into the prediction and runs away
    if (!use_odom_twist_ && !reanchored && dt > 0.0 && dt < 0.5) {
      const Eigen::Vector3d v_new = result.pose.rotation().transpose() *
                                    (result.pose.translation() - T_.translation()) / dt;
      Eigen::Vector3d dv = (1.0 - vel_smoothing_) * (v_new - v_body_);
      const double dv_max = max_accel_ * dt;
      if (dv.norm() > dv_max) dv *= dv_max / dv.norm();
      v_body_ += dv;
      if (v_body_.norm() > max_velocity_)
        v_body_ *= max_velocity_ / v_body_.norm();
    }
    T_ = result.pose;
    last_scan_end_ = scan.t_end;
    T_prop_ = T_;
    t_prop_ = scan.t_end;
    have_first_fix_ = true;

    publishOdom(T_, scan.t_end);
    if (publish_aligned_scan_) publishAligned(ds, scan.t_end);
    if (publish_2d_scan_ && scan_2d_pub_->get_subscription_count() > 0)
      publishBand2D(ds, scan.t_end);

    // detection runs only on a confident, locked fix — a mislocalized pose
    // would paint the whole scan as foreground. The divergence-coast path
    // already returned above; here we additionally require convergence and
    // skip the frame we just re-anchored on.
    if (detect_en_ && detector_ && have_first_fix_ && !reanchored &&
        result.converged) {
      runDetection(pts, scan.t_end);
    }

    if (print_stats_) {
      // STAT lines are machine-parseable (key=value) for offline analysis
      RCLCPP_INFO(get_logger(),
                  "STAT t=%.3f raw=%zu ds=%zu prep_ms=%.1f icp_ms=%.1f "
                  "iters=%d corr=%d conv=%d th=%.3f dev_t=%.3f dev_r=%.2f "
                  "dt=%.3f v=%.2f q=%zu lat_ms=%.0f",
                  scan.t_end, scan.points.size(), ds.size(), prep_ms, icp_ms,
                  result.iterations, result.num_correspondences,
                  result.converged ? 1 : 0, th, dev_t, dev_r * 180.0 / M_PI,
                  dt, v_body_.norm(), pending_.size(),
                  (nowSec() - scan.arrival_wall) * 1e3);
    }
  }

  // --------------------------- publishing ---------------------------
  // Project a pose onto the 2D-mode planar manifold: keep (x, y), pin z to
  // plane_z_, and set the attitude to Rz(yaw) * R_level_ — only yaw is free,
  // roll/pitch are FIXED to the LiDAR mount tilt. R_level_ levels the ground
  // normal (crop_n_ from ground_lidar.yaml) onto the map vertical, so a band
  // point projects as Rz(yaw)*(R_level_*p): leveled first, then yawed → correct
  // horizontal (x,y) for the SDF. The SDF observes none of z/roll/pitch; left
  // free they dead-reckon away (the pose floated ~+20 m of z over 60 s), and
  // pinning roll/pitch to 0 instead of the real tilt would mis-project the
  // tilted scan (a 0.25 m band at 15° smears walls ~6 cm).
  // extract the planar yaw of a pose, undoing the fixed mount tilt R_level_
  double planarYaw(const Eigen::Isometry3d &T) const {
    const Eigen::Matrix3d Ryaw = T.linear() * R_level_.transpose();
    return std::atan2(Ryaw(1, 0), Ryaw(0, 0));
  }

  // build a planar pose (x, y, plane_z_) with attitude Rz(yaw) * R_level_
  Eigen::Isometry3d fromPlanar(double x, double y, double yaw) const {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    P.linear() =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_level_;
    P.translation() = Eigen::Vector3d(x, y, plane_z_);
    return P;
  }

  Eigen::Isometry3d planarize(const Eigen::Isometry3d &T) const {
    return fromPlanar(T.translation().x(), T.translation().y(), planarYaw(T));
  }

  // Complementary low-pass for 2D mode: move a fraction `gain` from the smooth
  // gyro/CV prediction `pred` toward the noisy SDF fix `meas`. gain=1 takes the
  // fix outright (jittery); gain<1 lets the gyro carry the fast motion and only
  // slowly applies the absolute (drift) correction, killing per-scan yaw/xy
  // chatter. Not velocity damping (that adds prediction lag and worsens jitter)
  // — it low-passes only the absolute pose, so dynamic motion stays responsive.
  Eigen::Isometry3d blendPlanar(const Eigen::Isometry3d &pred,
                                const Eigen::Isometry3d &meas, double gain) const {
    const double xp = pred.translation().x(), yp = pred.translation().y();
    const double xm = meas.translation().x(), ym = meas.translation().y();
    const double yp_ = planarYaw(pred);
    const double dyaw = std::remainder(planarYaw(meas) - yp_, 2.0 * M_PI);
    return fromPlanar(xp + gain * (xm - xp), yp + gain * (ym - yp),
                      yp_ + gain * dyaw);
  }

  void publishOdom(const Eigen::Isometry3d &T_in, double t) {
    // 2D mode: never publish a floating/off-plane pose (z/roll/pitch unobserved)
    const Eigen::Isometry3d T = localization_2d_ ? planarize(T_in) : T_in;
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = base_frame_;
    const Eigen::Quaterniond q(T.rotation());
    odom.pose.pose.position.x = T.translation().x();
    odom.pose.pose.position.y = T.translation().y();
    odom.pose.pose.position.z = T.translation().z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.twist.twist.linear.x = v_body_.x();
    odom.twist.twist.linear.y = v_body_.y();
    odom.twist.twist.linear.z = v_body_.z();
    odom_pub_->publish(odom);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = T.translation().x();
      tf.transform.translation.y = T.translation().y();
      tf.transform.translation.z = T.translation().z();
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  void publishAligned(const std::vector<Eigen::Vector3d> &pts, double t) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(pts.size());
    for (const auto &p : pts) {
      const Eigen::Vector3d pw = T_ * p;
      cloud.emplace_back(pw.x(), pw.y(), pw.z());
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    aligned_pub_->publish(msg);
  }

  // 2D-ized aligned scan: the ground-band slab of `pts` (sensor frame) projected
  // into the map frame and FLATTENED to z=0 — the same planar (x,y) view the 2D
  // scan-to-SDF registration matches against, and the plane the /map track raster
  // (OccupancyGrid at z=0) is drawn on, so the two overlay exactly in RViz. Same
  // z-band as detection (crop_z_min/max above the GLIM ground plane). Runs
  // regardless of localization_2d; in 2D mode `pts` is already band-cropped.
  void publishBand2D(const std::vector<Eigen::Vector3d> &pts, double t) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(pts.size());
    for (const auto &p : pts) {
      const double hgt = crop_n_.dot(p) + crop_h_;  // height above ground (sensor frame)
      if (hgt < crop_z_min_ || hgt > crop_z_max_) continue;
      const Eigen::Vector3d pw = T_ * p;
      cloud.emplace_back(pw.x(), pw.y(), 0.0);  // flatten onto the map ground plane
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    scan_2d_pub_->publish(msg);
  }

  // --------------------------- detection ---------------------------
  // `scan_sensor` is the deskewed scan in the sensor frame; transform to the
  // map frame with the just-solved pose and feed the BEV detector.
  void runDetection(const std::vector<Eigen::Vector3d> &scan_sensor, double t) {
    std::vector<Eigen::Vector3d> pts_map;
    pts_map.reserve(scan_sensor.size());
    for (const auto &p : scan_sensor) pts_map.push_back(T_ * p);

    // GLIM ground plane (sensor frame) rotated into the map frame for this pose:
    // height(p_map) = n_map.p_map + h_map, identical to the sensor-frame height.
    const Eigen::Vector3d n_map = T_.rotation() * crop_n_;
    const double h_map = crop_h_ - n_map.dot(T_.translation());
    // ground z at (x,y): solve n_map.(x,y,z) + h_map = 0 for marker placement
    const double nz = (std::abs(n_map.z()) > 1e-6) ? n_map.z() : 1e-6;
    auto ground_z = [&](double x, double y) {
      return -(n_map.x() * x + n_map.y() * y + h_map) / nz;
    };

    const BevResult res = detector_->Update(pts_map, t, n_map, h_map);
    const auto stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));

    // foreground cloud: points after ground removal + track filter, i.e. the
    // BEV detector's input *before* DBSCAN clustering / tracking (on-track
    // obstacles + opponent kept, off-track / walls removed)
    if (obstacle_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<pcl::PointXYZ> oc;
      oc.reserve(res.obstacle_points.size());
      for (const auto &p : res.obstacle_points) oc.emplace_back(p.x(), p.y(), p.z());
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(oc, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = stamp;
      obstacle_pub_->publish(msg);
    }

    // cluster centers as PoseArray (map frame) — published every cycle
    // (empty list included) so consumers (e.g. MPCC) use overwrite semantics.
    geometry_msgs::msg::PoseArray poses;
    poses.header.frame_id = map_frame_;
    poses.header.stamp = stamp;

    // detection markers: bbox cube + id/speed label per cluster
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = map_frame_;
    clear.header.stamp = stamp;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    for (const auto &d : res.detections) {
      const double gz = ground_z(d.center.x(), d.center.y());

      geometry_msgs::msg::Pose pose;
      pose.position.x = d.center.x();
      pose.position.y = d.center.y();
      pose.position.z = gz;  // ground level; consumers typically use x,y only
      pose.orientation.w = 1.0;
      poses.poses.push_back(pose);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = stamp;
      m.ns = "objects";
      m.id = d.id;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = d.center.x();
      m.pose.position.y = d.center.y();
      m.pose.position.z = gz + bev_params_.z_min + 0.5 * std::max(0.1, d.height);
      m.pose.orientation.w = 1.0;
      m.scale.x = std::max(0.1, d.size.x());
      m.scale.y = std::max(0.1, d.size.y());
      m.scale.z = std::max(0.1, d.height);
      m.color.a = 0.4f;
      // moving (opponent) = red, static (wall/obstacle) = gray
      m.color.r = d.moving ? 1.0f : 0.6f;
      m.color.g = d.moving ? 0.1f : 0.6f;
      m.color.b = d.moving ? 0.1f : 0.6f;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      arr.markers.push_back(m);

      if (d.moving) {  // velocity arrow: direction = heading, length ~ speed
        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = map_frame_;
        arrow.header.stamp = stamp;
        arrow.ns = "velocity";
        arrow.id = d.id;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        const double az = gz + bev_params_.z_min + std::max(0.1, d.height) + 0.2;
        geometry_msgs::msg::Point p0, p1;
        p0.x = d.center.x();
        p0.y = d.center.y();
        p0.z = az;
        p1.x = d.center.x() + d.velocity.x() * arrow_scale_;
        p1.y = d.center.y() + d.velocity.y() * arrow_scale_;
        p1.z = az;
        arrow.points.push_back(p0);
        arrow.points.push_back(p1);
        arrow.scale.x = 0.05;  // shaft diameter
        arrow.scale.y = 0.12;  // head diameter
        arrow.scale.z = 0.15;  // head length
        arrow.color.a = 1.0f;
        arrow.color.r = 1.0f;
        arrow.color.g = 0.1f;
        arrow.color.b = 0.1f;
        arrow.lifetime = rclcpp::Duration::from_seconds(0.3);
        arr.markers.push_back(arrow);
      }
    }
    det_pub_->publish(arr);
    obstacle_pose_pub_->publish(poses);
  }

  // --------------------------- members ---------------------------
  // params
  std::string map_pcd_path_;  // active map (selected from 2d/3d by localization_2d)
  std::string map_pcd_3d_, map_pcd_2d_, ground_yaml_;
  std::string lidar_topic_, imu_topic_, map_frame_, base_frame_;
  // 2D LaserScan input front-end + wheel-odom translation source
  std::string input_mode_, scan_topic_, odom_twist_topic_;
  bool input_scan_ = false, use_odom_twist_ = false;
  double map_voxel_size_, scan_voxel_size_, min_range_, max_range_;
  int map_max_points_, point_filter_num_, max_iterations_, imu_init_samples_;
  double convergence_eps_, initial_threshold_, min_threshold_, min_motion_;
  double reject_trans_, reject_rot_deg_, max_velocity_, max_accel_;
  double adaptive_range_, vel_smoothing_;
  int reject_recover_count_ = 3;
  int consecutive_rejects_ = 0;
  bool stamp_at_scan_end_, imu_en_, deskew_en_, imu_rate_odom_,
      publish_tf_, publish_aligned_scan_, use_initial_pose_topic_, print_stats_,
      use_normals_;
  std::vector<double> initial_pose_;
  Eigen::Matrix3d R_il_ = Eigen::Matrix3d::Identity();
  // 2D localization (band-crop scan for ICP) + ground-band plane geometry
  bool localization_2d_ = false;
  Eigen::Vector3d crop_n_ = Eigen::Vector3d::UnitZ();
  double crop_h_ = 0.0, crop_z_min_ = 0.05, crop_z_max_ = 0.30;
  bool detect_en_ = false;
  std::string odom_topic_, obstacle_topic_, detection_topic_, obstacle_pose_topic_;
  // 2D occupancy grid (self-published, no nav2_map_server)
  bool publish_2d_map_ = true;
  std::string map_2d_topic_, map_2d_yaml_;
  nav_msgs::msg::OccupancyGrid grid_msg_;
  bool grid_ready_ = false;
  BevParams bev_params_;
  std::string track_map_path_;
  TrackMask track_mask_;       // detection stage-2 off-track filter
  std::string loc_map_path_;   // 2D-localization SDF map yaml (empty -> track map)
  TrackMask loc_mask_;         // 2D scan-to-SDF registration target
  double arrow_scale_ = 0.5;

  // map & estimation state
  VoxelHashMap map_;
  std::vector<Eigen::Vector3d> map_cloud_raw_;
  std::unique_ptr<AdaptiveThreshold> adaptive_;
  Eigen::Isometry3d T_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_prop_ = Eigen::Isometry3d::Identity();
  Eigen::Vector3d v_body_ = Eigen::Vector3d::Zero();
  double plane_z_ = 0.0;  // 2D-mode ground-plane height the pose is pinned to
  Eigen::Matrix3d R_level_ = Eigen::Matrix3d::Identity();  // 2D: levels crop_n_ -> map z
  double corr_gain_ = 1.0;  // 2D scan-to-SDF complementary-filter gain (1=raw fix)
  double last_scan_end_ = -1.0;
  double t_prop_ = -1.0;
  bool have_first_fix_ = false;

  // imu
  std::deque<ImuSample> imu_buf_;
  std::vector<Eigen::Vector3d> bias_acc_;
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  bool bias_ready_ = false;

  std::deque<PendingScan> pending_;

  // ros interfaces
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_, map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_2d_pub_;
  bool publish_2d_scan_ = true;
  std::string scan_2d_topic_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initpose_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // detection
  std::unique_ptr<BevDetector> detector_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr det_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obstacle_pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
};

}  // namespace kiss_loc

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kiss_loc::LocalizationNode>());
  rclcpp::shutdown();
  return 0;
}

#ifndef DETECT_H
#define DETECT_H

#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <f110_msgs/msg/obstacle_array.hpp>
#include <f110_msgs/msg/obstacle.hpp>

#include "frenet_conversion.h"
#include "grid_filter.h"

#include <vector>
#include <string>

class Obstacle
{
public:
  int id;
  double center_x;
  double center_y;
  double size;
  double theta;

  Obstacle(double x, double y, double size, double theta);
  double squaredDist(const Obstacle &other);
};

using Point2D = std::pair<double, double>;

class Detect : public rclcpp::Node
{
public:
  Detect();
  ~Detect();

private:
  // tf2 buffer + listener
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  tf2::Vector3 T_;
  tf2::Quaternion quat_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr global_wpnts_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_frenet_sub_;

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr breakpoints_markers_pub_;
  rclcpp::Publisher<f110_msgs::msg::ObstacleArray>::SharedPtr obstacles_msg_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr latency_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr on_track_points_pub_;

  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msgs;

  // Parameters (set via ROS params)
  double rate_;
  double lambda_angle_;  // in radians
  double sigma_;
  double new_cluster_threshold_m_;
  double min_size_m_;
  double min_2_points_dist_;
  int min_size_n_;
  int filter_kernel_size_;
  double max_size_m_;
  double max_viewing_distance_;
  double boundaries_inflation_;

  // Variables
  rclcpp::Time current_stamp_;
  std::vector<Obstacle> tracked_obstacles_;
  std::vector<std::vector<double>> wpnts_data_;
  std::vector<std::vector<double>> waypoints_;
  std::vector<double> s_array_;
  std::vector<double> d_right_array_;
  std::vector<double> d_left_array_;
  double track_length_;

  double car_s_;

  bool measuring_;
  bool from_bag_;
  bool path_needs_update_;
  std::string map_name_;
  std::string save_yaml_path_;
  frenet_conversion::FrenetConverter frenet_converter_;

  GridFilter GridFilter_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Dynamic-parameter callback handle (ROS2 replacement for dynamic_reconfigure)
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // Callbacks
  void laserCb(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void pathCb(const f110_msgs::msg::WpntArray::ConstSharedPtr msg);
  void carStateCb(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  rcl_interfaces::msg::SetParametersResult dynParamCb(
      const std::vector<rclcpp::Parameter> &params);

  // Timer callback
  void timerCallback();

  // Utility functions
  double normalizeS(double x, double track_length);
  bool laserPointOnTrack(double s, double d);
  // declare a numeric param that accepts int OR double from yaml, return as double
  double declareNumber(const std::string &name, double default_value);
  void saveYaml();  // write the detect: block back to opponent_tracker_params.yaml
  void publishBreakpoints(const std::vector<std::vector<std::pair<double, double>>> &objects_pointcloud_list);

  visualization_msgs::msg::MarkerArray clearmarkers();

  // Processing functions (clustering, obstacle fitting, etc.)
  std::vector<std::vector<std::pair<double, double>>> clustering(const sensor_msgs::msg::LaserScan::ConstSharedPtr &msg);
  std::vector<Obstacle> fittingLShape(const std::vector<std::vector<std::pair<double, double>>> &objects_pointcloud_list);
  void checkObstacles(std::vector<Obstacle> &current_obstacles);
  void publishObstaclesMessage();
  void publishObstaclesMarkers();
  void publishOnTrackPointCloud(const std::vector<Point2D> &on_track_points);

  // Converter initialization
  void initializeConverter();
};

#endif // DETECT_H

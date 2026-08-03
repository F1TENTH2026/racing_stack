#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kiss_loc {

inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
      -v.y(), v.x(), 0.0;
  return m;
}

inline Eigen::Matrix3d So3Exp(const Eigen::Vector3d &w) {
  const double theta = w.norm();
  if (theta < 1e-12) return Eigen::Matrix3d::Identity() + Skew(w);
  return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

inline double RotationAngle(const Eigen::Matrix3d &R) {
  return Eigen::AngleAxisd(R).angle();
}

}  // namespace kiss_loc

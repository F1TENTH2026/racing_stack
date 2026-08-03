#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

#include "kiss_icp_localization/se3.hpp"

namespace kiss_loc {

// KISS-ICP adaptive threshold: the correspondence search radius is the
// running RMS of the model (prediction) error. Errors below
// min_motion are ignored, so the threshold never collapses to zero.
class AdaptiveThreshold {
public:
  AdaptiveThreshold(double initial_threshold, double min_threshold,
                    double min_motion, double max_range)
      : initial_threshold_(initial_threshold),
        min_threshold_(min_threshold),
        min_motion_(min_motion),
        max_range_(max_range) {}

  double ComputeThreshold() const {
    if (num_samples_ < 1) return initial_threshold_;
    return std::max(min_threshold_, std::sqrt(model_sse_ / num_samples_));
  }

  // deviation = prediction^{-1} * registered_pose
  void UpdateModelDeviation(const Eigen::Isometry3d &deviation) {
    const double theta = RotationAngle(deviation.rotation());
    const double err =
        deviation.translation().norm() + 2.0 * max_range_ * std::sin(theta / 2.0);
    if (err > min_motion_) {
      model_sse_ += err * err;
      ++num_samples_;
    }
  }

  void Reset() {
    model_sse_ = 0.0;
    num_samples_ = 0;
  }

private:
  double initial_threshold_;
  double min_threshold_;
  double min_motion_;
  double max_range_;
  double model_sse_ = 0.0;
  int num_samples_ = 0;
};

}  // namespace kiss_loc

#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "kiss_icp_localization/track_mask.hpp"
#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace kiss_loc {

struct RegistrationResult {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  int num_correspondences = 0;
  int iterations = 0;
  bool converged = false;
};

// Robust ICP (Gauss-Newton, Geman-McClure kernel) of a scan (sensor frame)
// against the static map, starting from initial_guess. Uses point-to-plane
// residuals where the map carries normals (and use_normals is set),
// point-to-point otherwise.
RegistrationResult AlignScanToMap(const std::vector<Eigen::Vector3d> &scan,
                                  const VoxelHashMap &map,
                                  const Eigen::Isometry3d &initial_guess,
                                  double max_corr_dist, double kernel_sigma,
                                  int max_iterations, double convergence_eps,
                                  bool use_normals = false);

// 2D scan-to-SDF registration for planar (2D) localization. The scan (sensor
// frame) is matched against the track-mask signed distance field: each point's
// residual is the SDF value at its map-frame (x, y) projection (zero on a wall),
// minimized with a Geman-McClure kernel. Only the planar pose (x, y, yaw) is
// optimized; z/roll/pitch are carried unchanged from initial_guess, so the
// returned pose is a full SE(3) isometry consumable by the same downstream
// pipeline as AlignScanToMap. max_corr_dist gates points by |SDF| (points far
// from any wall are ignored). result.num_correspondences counts points that
// landed on the SDF grid this iteration.
RegistrationResult AlignScanToTrackSdf(const std::vector<Eigen::Vector3d> &scan,
                                       const TrackMask &track,
                                       const Eigen::Isometry3d &initial_guess,
                                       double max_corr_dist, double kernel_sigma,
                                       int max_iterations,
                                       double convergence_eps);

}  // namespace kiss_loc

#include "kiss_icp_localization/voxel_hash_map.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>

namespace kiss_loc {

namespace {
inline int VoxelCoord(double v, double voxel_size) {
  return static_cast<int>(std::floor(v / voxel_size));
}
}  // namespace

void VoxelHashMap::Build(const std::vector<Eigen::Vector3d> &points,
                         const std::vector<Eigen::Vector3d> &normals) {
  grid_.clear();
  num_points_ = 0;
  has_normals_ = normals.size() == points.size() && !normals.empty();
  grid_.reserve(points.size() / 4 + 1);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto &p = points[i];
    const uint64_t key = Key(VoxelCoord(p.x(), voxel_size_),
                             VoxelCoord(p.y(), voxel_size_),
                             VoxelCoord(p.z(), voxel_size_));
    auto &bucket = grid_[key];
    if (static_cast<int>(bucket.size()) < max_points_per_voxel_) {
      Eigen::Vector3d n = Eigen::Vector3d::Zero();
      if (has_normals_ && normals[i].allFinite() && normals[i].norm() > 0.5)
        n = normals[i].normalized();
      bucket.push_back({p, n});
      ++num_points_;
    }
  }
}

void VoxelHashMap::EstimateNormals() {
  if (grid_.empty()) return;
  std::vector<std::vector<MapPoint> *> buckets;
  buckets.reserve(grid_.size());
  for (auto &kv : grid_) buckets.push_back(&kv.second);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
  for (size_t i = 0; i < buckets.size(); ++i) {
    auto &bucket = *buckets[i];
    const Eigen::Vector3d &p0 = bucket.front().p;
    const int cx = VoxelCoord(p0.x(), voxel_size_);
    const int cy = VoxelCoord(p0.y(), voxel_size_);
    const int cz = VoxelCoord(p0.z(), voxel_size_);

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sq = Eigen::Matrix3d::Zero();
    int n = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const auto it = grid_.find(Key(cx + dx, cy + dy, cz + dz));
          if (it == grid_.end()) continue;
          for (const auto &mp : it->second) {
            mean += mp.p;
            sq.noalias() += mp.p * mp.p.transpose();
            ++n;
          }
        }
      }
    }
    if (n < 8) continue;
    mean /= n;
    const Eigen::Matrix3d cov = sq / n - mean * mean.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    const Eigen::Vector3d &ev = es.eigenvalues();  // ascending
    // planar enough: the smallest spread is clearly below the others
    if (ev(0) > 0.25 * ev(1)) continue;
    const Eigen::Vector3d normal = es.eigenvectors().col(0);
    for (auto &mp : bucket) mp.n = normal;
  }
  has_normals_ = true;
}

bool VoxelHashMap::GetClosestNeighbor(const Eigen::Vector3d &query,
                                      double max_dist,
                                      MapPoint &closest) const {
  const int cx = VoxelCoord(query.x(), voxel_size_);
  const int cy = VoxelCoord(query.y(), voxel_size_);
  const int cz = VoxelCoord(query.z(), voxel_size_);
  // Adjacent shell only (27 voxels), as in KISS-ICP — searching wider for
  // large thresholds makes the per-frame cost explode
  constexpr int radius = 1;

  double best_d2 = max_dist * max_dist;
  bool found = false;
  for (int dx = -radius; dx <= radius; ++dx) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dz = -radius; dz <= radius; ++dz) {
        const auto it = grid_.find(Key(cx + dx, cy + dy, cz + dz));
        if (it == grid_.end()) continue;
        for (const auto &mp : it->second) {
          const double d2 = (mp.p - query).squaredNorm();
          if (d2 < best_d2) {
            best_d2 = d2;
            closest = mp;
            found = true;
          }
        }
      }
    }
  }
  return found;
}

std::vector<Eigen::Vector3d> VoxelDownsample(
    const std::vector<Eigen::Vector3d> &points, double voxel_size) {
  std::unordered_map<uint64_t, Eigen::Vector3d> grid;
  grid.reserve(points.size() / 2 + 1);
  constexpr uint64_t kOffset = 1u << 20;
  std::vector<Eigen::Vector3d> out;
  out.reserve(points.size() / 2 + 1);
  for (const auto &p : points) {
    const uint64_t key =
        ((static_cast<uint64_t>(VoxelCoord(p.x(), voxel_size)) + kOffset) << 42) |
        ((static_cast<uint64_t>(VoxelCoord(p.y(), voxel_size)) + kOffset) << 21) |
        (static_cast<uint64_t>(VoxelCoord(p.z(), voxel_size)) + kOffset);
    if (grid.emplace(key, p).second) out.push_back(p);
  }
  return out;
}

}  // namespace kiss_loc

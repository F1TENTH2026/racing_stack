#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kiss_loc {

struct MapPoint {
  Eigen::Vector3d p;
  Eigen::Vector3d n;  // surface normal; zero when unknown
};

// Static voxel-hash map over the prior map cloud. KISS-ICP style:
// fixed-size voxels, capped points per voxel, nearest-neighbor lookup
// over the query's voxel plus its adjacent shell.
class VoxelHashMap {
public:
  VoxelHashMap() = default;
  VoxelHashMap(double voxel_size, int max_points_per_voxel)
      : voxel_size_(voxel_size), max_points_per_voxel_(max_points_per_voxel) {}

  // normals may be empty (point-to-point only) or parallel to points
  void Build(const std::vector<Eigen::Vector3d> &points,
             const std::vector<Eigen::Vector3d> &normals = {});

  // Estimate per-voxel normals by PCA over each voxel and its adjacent
  // shell. Voxels that don't look planar keep zero normals (point-to-point
  // fallback). One-time cost at map load.
  void EstimateNormals();

  // Closest map point to `query` within `max_dist`. Search radius is the
  // adjacent voxel shell (27 voxels), as in KISS-ICP.
  bool GetClosestNeighbor(const Eigen::Vector3d &query, double max_dist,
                          MapPoint &closest) const;

  bool Empty() const { return grid_.empty(); }
  std::size_t NumPoints() const { return num_points_; }
  double VoxelSize() const { return voxel_size_; }
  bool HasNormals() const { return has_normals_; }

private:
  uint64_t Key(int x, int y, int z) const {
    constexpr uint64_t kOffset = 1u << 20;  // voxel indices must fit in ±2^20
    return ((static_cast<uint64_t>(x) + kOffset) << 42) |
           ((static_cast<uint64_t>(y) + kOffset) << 21) |
           (static_cast<uint64_t>(z) + kOffset);
  }

  double voxel_size_ = 0.5;
  int max_points_per_voxel_ = 30;
  std::size_t num_points_ = 0;
  bool has_normals_ = false;
  std::unordered_map<uint64_t, std::vector<MapPoint>> grid_;
};

// Keep the first point falling into each voxel (KISS-ICP downsampling).
std::vector<Eigen::Vector3d> VoxelDownsample(
    const std::vector<Eigen::Vector3d> &points, double voxel_size);

}  // namespace kiss_loc

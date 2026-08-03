#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kiss_icp_localization/track_mask.hpp"

namespace kiss_loc {

// Per-frame BEV obstacle detector. Using the localized (map-frame) scan it
// removes the ground via the known map ground plane, crops to a vehicle-height
// band, and drops points outside the track mask (drivable area). The survivors
// are obstacles / the opponent on the track; they are projected to a 2D grid,
// clustered, and tracked across frames so each gets a velocity / moving flag
// (moving => opponent).
//
// Stateless per frame (no rolling background): nothing to warm up, no
// leading-edge false positives as the platform drives into new area, and
// insensitive to small pose jitter. Consumes the already-deskewed, already-
// localized scan, so it adds no registration. A moved/new static object on the
// track is reported as an obstacle — the desired behavior.
struct BevParams {
  double res = 0.2;          // BEV cell size [m]
  // Ground plane comes from ground_lidar.yaml (GLIM): the same sensor-frame
  // plane used to crop the localization input scan. It is passed to Update()
  // already transformed into the map frame for the just-solved pose, so the
  // height of a point p is ground_normal.p + ground_offset. The z-band below
  // matches crop_z_min/crop_z_max from that yaml.
  double z_min = 0.05;       // keep points this far above ground [m] (ground removal)
  double z_max = 0.30;       // ... and below this (drop overhead)
  // Track filter: drop points by signed distance to the track mask boundary
  // (TrackMask, GLIM map_track of the drivable area). A point is dropped when
  // SignedOutside(x,y) > track_margin. This is the in-plane companion to the
  // normal-direction z-band crop above and is the only spatial filter: track
  // inside = obstacle (kept), outside = wall / off-track (removed).
  bool track_filter = false; // require points to lie on/near the track mask
  double track_margin = 0.0; // signed: >0 dilate (tolerant), <0 erode (drop near-wall band) [m]
  // DBSCAN over occupied cells, independent of cell size `res`: eps is the
  // neighborhood radius and a cell needs >= min_samples occupied cells (incl.
  // itself) within eps to be a core. Sparse cells become noise and are dropped,
  // which rejects spurious returns. Cost grows as (eps/res)^2.
  double eps = 0.2;          // DBSCAN neighborhood radius [m]
  int min_samples = 4;       // min cells within eps (incl. self) for a core cell
  int min_cluster_cells = 2; // final: discard clusters smaller than this (cells)
  double track_gate = 1.0;   // max bbox-center step to associate a track [m]
  double moving_speed = 0.5; // |v| above this  => moving (opponent) [m/s]
  int max_misses = 5;        // drop a track after this many unmatched frames
};

struct Detection {
  int id = -1;
  Eigen::Vector2d center{0.0, 0.0};    // bbox center, map frame [m]
  Eigen::Vector2d size{0.0, 0.0};      // bbox extent x,y [m]
  Eigen::Vector2d velocity{0.0, 0.0};  // map frame [m/s]
  double speed = 0.0;
  bool moving = false;
  int num_points = 0;
  int num_cells = 0;
  double height = 0.0;  // cluster height above ground [m]
};

struct BevResult {
  std::vector<Detection> detections;
  std::vector<Eigen::Vector3d> obstacle_points;  // map-frame band-cropped points
};

class BevDetector {
public:
  // track: 2D track mask (may be null when track_filter off, or invalid if the
  // mask failed to load — filtering is then skipped).
  explicit BevDetector(const BevParams &p, const TrackMask *track = nullptr)
      : p_(p), track_(track) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for track velocity.
  // ground_normal / ground_offset: ground plane in the map frame (the GLIM
  // sensor-frame plane from ground_lidar.yaml rotated by the current pose);
  // height above ground of a point p is ground_normal.dot(p) + ground_offset.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp,
                   const Eigen::Vector3d &ground_normal, double ground_offset);

private:
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  BevParams p_;
  const TrackMask *track_ = nullptr;

  struct Track {
    int id;
    Eigen::Vector2d c;
    Eigen::Vector2d v;
    double last;
    int misses;
  };
  std::vector<Track> tracks_;
  int next_id_ = 0;
};

}  // namespace kiss_loc

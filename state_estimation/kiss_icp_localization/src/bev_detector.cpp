#include "kiss_icp_localization/bev_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace kiss_loc {

namespace {
struct Obs {
  int n = 0;
  float rmin = std::numeric_limits<float>::max();
  float rmax = -std::numeric_limits<float>::max();
  int label = 0;  // DBSCAN: 0=undefined, -1=noise, >0=cluster id
};
}  // namespace

BevResult BevDetector::Update(const std::vector<Eigen::Vector3d> &points_map,
                              double stamp,
                              const Eigen::Vector3d &ground_normal,
                              double ground_offset) {
  BevResult out;
  const double inv_res = 1.0 / p_.res;

  const bool do_track = p_.track_filter && track_ && track_->Valid();

  // ---- ground removal + height crop + track filter, to 2D cells ----
  std::unordered_map<int64_t, Obs> cells;
  cells.reserve(points_map.size() / 4 + 16);
  out.obstacle_points.reserve(points_map.size() / 4 + 16);
  for (const auto &p : points_map) {
    const double r = ground_normal.dot(p) + ground_offset;  // height above ground
    if (r < p_.z_min || r > p_.z_max) continue;  // ground / overhead removed
    // drop points by signed distance to the track boundary. margin>0 keeps an
    // outer ring (tolerant); margin<0 erodes a near-wall band (drops wall returns
    // that scatter just inside the track).
    if (do_track && track_->SignedOutside(p.x(), p.y()) > p_.track_margin)
      continue;
    const int ix = static_cast<int>(std::floor(p.x() * inv_res));
    const int iy = static_cast<int>(std::floor(p.y() * inv_res));
    Obs &o = cells[CellKey(ix, iy)];
    ++o.n;
    o.rmin = std::min(o.rmin, static_cast<float>(r));
    o.rmax = std::max(o.rmax, static_cast<float>(r));
    out.obstacle_points.push_back(p);
  }

  // ---- DBSCAN over occupied cells (density-based) ----
  // A cell is "core" if >= min_samples occupied cells (incl. itself) lie within
  // eps. Clusters grow only through core cells; cells reachable from a core are
  // border, the rest are noise and dropped. The res grid is the spatial index:
  // an eps-neighborhood is a +/-kr cell box (filtered to a true Euclidean disk).
  const int kr =
      std::max(1, std::min(20, static_cast<int>(std::ceil(p_.eps * inv_res))));
  const double eps2_cells = (p_.eps * inv_res) * (p_.eps * inv_res);
  auto decode = [](int64_t k, int &ix, int &iy) {
    ix = static_cast<int>(k >> 32);
    iy = static_cast<int>(static_cast<int32_t>(k & 0xffffffff));
  };
  auto range_query = [&](int64_t k, std::vector<int64_t> &out) {
    out.clear();
    int ix, iy;
    decode(k, ix, iy);
    for (int dx = -kr; dx <= kr; ++dx)
      for (int dy = -kr; dy <= kr; ++dy) {
        if (dx == 0 && dy == 0) continue;
        if (dx * dx + dy * dy > eps2_cells) continue;
        const int64_t nk = CellKey(ix + dx, iy + dy);
        if (cells.count(nk)) out.push_back(nk);
      }
  };

  int cid = 0;
  std::vector<int64_t> nb, seeds;
  for (auto &kv : cells) {
    if (kv.second.label != 0) continue;  // already visited
    range_query(kv.first, nb);
    if (static_cast<int>(nb.size()) + 1 < p_.min_samples) {
      kv.second.label = -1;  // provisional noise (may be claimed as border)
      continue;
    }
    ++cid;
    kv.second.label = cid;
    seeds = nb;
    for (size_t qi = 0; qi < seeds.size(); ++qi) {
      Obs &oq = cells.at(seeds[qi]);
      if (oq.label == -1) oq.label = cid;  // noise -> border of this cluster
      if (oq.label != 0) continue;         // already assigned/visited
      oq.label = cid;
      range_query(seeds[qi], nb);
      if (static_cast<int>(nb.size()) + 1 >= p_.min_samples)  // core -> expand
        for (const int64_t nk : nb) seeds.push_back(nk);
    }
  }

  // ---- aggregate clusters into detections ----
  struct Agg {
    int n = 0, cells = 0;
    int ixmin = INT32_MAX, ixmax = INT32_MIN, iymin = INT32_MAX, iymax = INT32_MIN;
    float hmin = std::numeric_limits<float>::max();
    float hmax = -std::numeric_limits<float>::max();
  };
  std::unordered_map<int, Agg> agg;
  for (const auto &kv : cells) {
    if (kv.second.label <= 0) continue;  // skip noise/undefined
    int ix, iy;
    decode(kv.first, ix, iy);
    Agg &a = agg[kv.second.label];
    a.n += kv.second.n;
    ++a.cells;
    a.ixmin = std::min(a.ixmin, ix);
    a.ixmax = std::max(a.ixmax, ix);
    a.iymin = std::min(a.iymin, iy);
    a.iymax = std::max(a.iymax, iy);
    a.hmin = std::min(a.hmin, kv.second.rmin);
    a.hmax = std::max(a.hmax, kv.second.rmax);
  }
  std::vector<Detection> raw;
  for (const auto &ak : agg) {
    const Agg &a = ak.second;
    if (a.cells < p_.min_cluster_cells) continue;
    Detection d;
    const double x0 = a.ixmin * p_.res, x1 = (a.ixmax + 1) * p_.res;
    const double y0 = a.iymin * p_.res, y1 = (a.iymax + 1) * p_.res;
    d.center = Eigen::Vector2d(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    d.size = Eigen::Vector2d(x1 - x0, y1 - y0);
    d.num_points = a.n;
    d.num_cells = a.cells;
    d.height = static_cast<double>(a.hmax - a.hmin);
    raw.push_back(d);
  }

  // ---- track association (greedy nearest bbox center) ----
  std::vector<bool> track_used(tracks_.size(), false);
  for (auto &d : raw) {
    int best = -1;
    double best_d2 = p_.track_gate * p_.track_gate;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (track_used[i]) continue;
      const double d2 = (tracks_[i].c - d.center).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(i);
      }
    }
    if (best >= 0) {
      Track &t = tracks_[best];
      const double dt = std::max(1e-3, stamp - t.last);
      const Eigen::Vector2d v_inst = (d.center - t.c) / dt;
      t.v = 0.6 * t.v + 0.4 * v_inst;  // EMA
      t.c = d.center;
      t.last = stamp;
      t.misses = 0;
      track_used[best] = true;
      d.id = t.id;
      d.velocity = t.v;
    } else {
      Track t;
      t.id = next_id_++;
      t.c = d.center;
      t.v.setZero();
      t.last = stamp;
      t.misses = 0;
      tracks_.push_back(t);
      track_used.push_back(true);
      d.id = t.id;
    }
    d.speed = d.velocity.norm();
    d.moving = d.speed > p_.moving_speed;
  }

  // age out unmatched tracks
  for (size_t i = 0; i < tracks_.size();) {
    if (!track_used[i]) ++tracks_[i].misses;
    if (tracks_[i].misses > p_.max_misses) {
      tracks_[i] = tracks_.back();
      track_used[i] = track_used.back();
      tracks_.pop_back();
      track_used.pop_back();
    } else {
      ++i;
    }
  }

  out.detections = std::move(raw);
  return out;
}

}  // namespace kiss_loc

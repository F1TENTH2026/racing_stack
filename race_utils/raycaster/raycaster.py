#!/usr/bin/env python3
"""
RaycastEngine — one 2D LiDAR raycaster for BOTH the simulator and the particle filter.

Backends
  'rm'    : vendored range_libc C++ RayMarching (exact DT sphere-tracing). ★ DEFAULT.
            Fast and exact; needs the range_libc extension (installed via
            environment.yml / `pip install -e .../range_libc/pywrapper`).
  'pcddt' | 'cddt' | 'glt' | 'bl' | 'rmgpu' : other range_libc C++ backends.
  'lut'   : precomputed numpy table (built from the numba f1tenth_gym oracle).
            Pure-numpy at query time → NO range_libc needed (portable fallback),
            but LUT build is slow under the numba shim, so prefer 'rm' here.

Consumers
  Simulator      : scan(pose_xyt, num_beams, fov)              -> ranges[num_beams]  (m)
  Particle filter: calc_range_repeat_angles(particles, angles) -> ranges[M*K]        (m)

Coordinate convention (ROS): occupancy[row, col], row 0 at world y = origin_y
(bottom-origin). world->pixel: col=(x-ox)/res, row=(y-oy)/res. Use
RaycastEngine.load_map_yaml() to load an F1TENTH map image (it flips top-bottom).
"""
import os
import sys
import importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDORED_RL = os.path.join(_HERE, "range_libc", "pywrapper")
if os.path.isdir(_VENDORED_RL) and _VENDORED_RL not in sys.path:
    sys.path.insert(0, _VENDORED_RL)               # vendored range_libc/*.so

_RL_BACKENDS = ("glt", "pcddt", "cddt", "rm", "bl", "rmgpu")


def _load_numba_sim():
    """Import the vendored numba ScanSimulator2D (f1tenth_gym laser_models)."""
    p = os.path.join(_HERE, "vendor", "f110_laser_models.py")
    spec = importlib.util.spec_from_file_location("f110_laser_models", p)
    m = importlib.util.module_from_spec(spec); sys.modules["f110_laser_models"] = m
    spec.loader.exec_module(m)
    return m


class RaycastEngine:
    def __init__(self, backend="rm", max_range_m=10.0, theta_disc=360):
        self.backend = backend
        self.max_range_m = float(max_range_m)
        self.theta_disc = int(theta_disc)
        self.res = self.ox = self.oy = None
        self.W = self.H = None
        self._rl = None
        self._lut = None                            # [W,H,TD] float16 ranges in METERS
        self._amin = -np.pi
        self._da = 2 * np.pi / (self.theta_disc - 1)

    # ---------- map ----------
    def set_map(self, occupancy, resolution, origin_xy):
        occ = np.ascontiguousarray(occupancy).astype(bool)
        self.H, self.W = occ.shape
        self.res = float(resolution); self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        if self.backend in _RL_BACKENDS:
            import range_libc
            mrpx = self.max_range_m / self.res
            # fixed numpy ctor: sets the world transform → query in world meters
            omap = range_libc.PyOMap(occ, resolution=self.res, origin_x=self.ox, origin_y=self.oy)
            if self.backend == "glt":
                self._rl = range_libc.PyGiantLUTCast(omap, mrpx, self.theta_disc)
            elif self.backend in ("cddt", "pcddt"):
                self._rl = range_libc.PyCDDTCast(omap, mrpx, self.theta_disc)
                if self.backend == "pcddt":
                    self._rl.prune()
            elif self.backend == "rm":
                self._rl = range_libc.PyRayMarching(omap, mrpx)
            elif self.backend == "rmgpu":
                self._rl = range_libc.PyRayMarchingGPU(omap, mrpx)   # needs WITH_CUDA=ON build
            elif self.backend == "bl":
                self._rl = range_libc.PyBresenhamsLine(omap, mrpx)
        elif self.backend == "lut":
            self._materialize_lut(occ)
        else:
            raise ValueError(f"unknown backend {self.backend}")
        return self

    def _w2p(self, x, y):
        return (x - self.ox) / self.res, (y - self.oy) / self.res

    # ---------- SIMULATOR API ----------
    @staticmethod
    def _apply_miss(out, mr, miss):
        """Real lidars report no-return (not max_range) when a beam hits nothing.
        miss=None keeps the clamped max_range (PF/benchmark default); miss=nan/inf
        marks un-returned beams so they are not drawn as a false wall at max_range."""
        if miss is None:
            return out
        # tol = 1 cell: precompute backends (glt/cddt) return ~max_range-epsilon (not exactly mr)
        # for un-returned beams, so a tight threshold would still draw them as a false wall.
        return np.where(out >= mr - 1e-3, miss, out).astype(np.float64)

    def scan(self, pose_xyt, num_beams, fov, max_range=None, miss=None):
        mr = max_range or self.max_range_m
        phi = pose_xyt[2] + np.linspace(-fov / 2, fov / 2, num_beams)
        if self.backend == "lut":
            out = np.minimum(self._lut_lookup(pose_xyt[0], pose_xyt[1], phi), mr)
            return self._apply_miss(out, mr, miss)
        # range_libc (fixed ctor): query world (x, y, theta) directly → meters out
        q = np.empty((num_beams, 3), np.float32)
        q[:, 0] = pose_xyt[0]; q[:, 1] = pose_xyt[1]; q[:, 2] = phi
        out = np.zeros(num_beams, np.float32)
        self._rl.calc_range_many(np.ascontiguousarray(q), out)
        return self._apply_miss(np.minimum(out, mr), mr, miss)

    # ---------- PARTICLE FILTER API ----------
    def calc_range_repeat_angles(self, particles_xyt, angles):
        M, K = particles_xyt.shape[0], angles.shape[0]
        if self.backend == "lut":
            phi = particles_xyt[:, 2][:, None] + angles[None, :]               # [M,K]
            xi, yi = self._w2p(particles_xyt[:, 0], particles_xyt[:, 1])
            xi = np.clip(xi.astype(np.int64), 0, self.W - 1)[:, None]
            yi = np.clip(yi.astype(np.int64), 0, self.H - 1)[:, None]
            k = self._abin(phi)
            return np.minimum(self._lut[xi, yi, k].astype(np.float32), self.max_range_m).reshape(M * K)
        parts = np.ascontiguousarray(particles_xyt[:, :3], np.float32)   # world (x,y,theta)
        out = np.zeros(M * K, np.float32)
        self._rl.calc_range_repeat_angles(parts, np.ascontiguousarray(angles, np.float32), out)
        return out                                                       # meters

    # ---------- dynamic overlays (NO map rebuild — works with precompute backends) ----------
    @staticmethod
    def _vertices(pose, length, width):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        loc = np.array([[length/2, width/2], [length/2, -width/2],
                        [-length/2, -width/2], [-length/2, width/2]])
        return loc @ np.array([[c, s], [-s, c]]) + np.asarray(pose[:2])   # [4,2] world

    def scan_with_dynamics(self, pose, num_beams, fov, opp_poses=None,
                           opp_size=(0.58, 0.31), obstacles=None, max_range=None, miss=np.nan):
        """Static precomputed scan + ray-cast overlay of opponents (rotated boxes)
        and obstacles (axis-aligned squares, [K,3] = x,y,half_side). Per beam: min(static, dynamic).
        Same idea f1tenth_gym uses for opponents — keeps GLT/CDDT/LUT precompute intact.
        miss (default nan) marks beams that returned nothing within max_range."""
        mr = max_range or self.max_range_m
        scan = self.scan(pose, num_beams, fov, mr).astype(np.float64)   # max_range on miss; nan applied at end
        ang = pose[2] + np.linspace(-fov / 2, fov / 2, num_beams)
        dx, dy = np.cos(ang), np.sin(ang)
        px, py = float(pose[0]), float(pose[1])
        if obstacles is not None and len(obstacles):                      # ray vs SQUARE (axis-aligned, side 2r)
            for cx, cy, r in np.asarray(obstacles, float):
                scan = self._overlay_box(scan, (cx, cy, 0.0), 2 * r, 2 * r, px, py, dx, dy)
        if opp_poses is not None and len(opp_poses):                      # ray vs oriented box (opponent car)
            for op in np.asarray(opp_poses, float):
                scan = self._overlay_box(scan, op, opp_size[0], opp_size[1], px, py, dx, dy)
        return self._apply_miss(np.minimum(scan, mr), mr, miss)

    def _overlay_box(self, scan, box_pose, length, width, px, py, dx, dy):
        """min(scan, distance to a rotated rectangle) per beam — ray vs 4 box edges."""
        V = self._vertices(box_pose, length, width); W = np.roll(V, -1, 0)
        for a, e in zip(V, W - V):
            det = e[0] * dy - e[1] * dx
            safe = np.abs(det) > 1e-12
            den = np.where(safe, det, 1.0)
            wx, wy = a[0] - px, a[1] - py
            t = (e[0] * wy - e[1] * wx) / den
            u = (dx * wy - dy * wx) / den
            ok = safe & (t > 0) & (u >= 0) & (u <= 1) & (t < scan)
            scan = np.where(ok, t, scan)
        return scan

    # ---------- LUT internals ----------
    def _abin(self, phi):
        w = (phi - self._amin) % (2 * np.pi) + self._amin
        return np.clip(np.rint((w - self._amin) / self._da).astype(np.int64), 0, self.theta_disc - 1)

    def _lut_lookup(self, x, y, phi):
        xi = int(np.clip((x - self.ox) / self.res, 0, self.W - 1))
        yi = int(np.clip((y - self.oy) / self.res, 0, self.H - 1))
        return self._lut[xi, yi, self._abin(np.asarray(phi))].astype(np.float32)

    def _materialize_lut(self, occ):
        """Build [W,H,theta_disc] range table (meters) from the numba oracle so it
        matches the existing simulator's frame exactly."""
        vlm = _load_numba_sim()
        TD = self.theta_disc
        self._amin, self._da = -np.pi, 2 * np.pi / (TD - 1)
        orc = vlm.ScanSimulator2D(TD, 2 * np.pi, max_range=self.max_range_m)
        orc.set_map_from_array(np.where(occ, 0, 255).astype(np.uint8),
                               self.res, self.ox, self.oy, 0.0)
        orc.scan(np.array([self.ox, self.oy, 0.0]), None)        # jit warmup
        lut = np.empty((self.W, self.H, TD), np.float16)
        pose = np.zeros(3)
        for col in range(self.W):
            pose[0] = self.ox + col * self.res
            for row in range(self.H):
                pose[1] = self.oy + row * self.res
                lut[col, row, :] = orc.scan(pose, None)
        self._lut = lut

    def save_lut(self, path):
        np.savez_compressed(path, lut=self._lut, resolution=self.res,
                            origin=np.array([self.ox, self.oy], np.float64),
                            max_range_m=self.max_range_m, theta_disc=self.theta_disc)

    @classmethod
    def load_lut(cls, path):
        """Fast path: load a precomputed LUT — NO range_libc / numba needed."""
        z = np.load(path)
        e = cls(backend="lut", max_range_m=float(z["max_range_m"]), theta_disc=int(z["theta_disc"]))
        e._lut = z["lut"]; e.W, e.H = e._lut.shape[0], e._lut.shape[1]
        e.res = float(z["resolution"]); e.ox, e.oy = [float(v) for v in z["origin"]]
        e._amin, e._da = -np.pi, 2 * np.pi / (e.theta_disc - 1)
        return e

    # ---------- helper: load an F1TENTH map yaml ----------
    @staticmethod
    def load_map_yaml(yaml_path):
        """-> (occupancy[H,W] bool row0=bottom, resolution, (origin_x, origin_y))."""
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_path))
        img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta["image"])))
        if img.ndim == 3:
            img = img[..., 0]
        occ = np.flipud(img) <= 128
        return occ, float(meta["resolution"]), (meta["origin"][0], meta["origin"][1])


if __name__ == "__main__":
    import time
    CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
    occ, res, origin = RaycastEngine.load_map_yaml(CAC + "/stack_master/maps/f/f.yaml")
    ys, xs = np.where(~occ); i = len(xs) // 2
    pose = np.array([origin[0] + xs[i] * res, origin[1] + ys[i] * res, 0.3])
    print(f"map {occ.shape}, pose {pose.round(2)}")
    e = RaycastEngine(backend="lut", max_range_m=10.0, theta_disc=360).set_map(occ, res, origin)
    s = e.scan(pose, 1080, 4.7)
    parts = np.tile(pose, (4000, 1)).astype(np.float32); ang = np.linspace(-2.35, 2.35, 100).astype(np.float32)
    t = time.perf_counter()
    for _ in range(20): e.calc_range_repeat_angles(parts, ang)
    pf = (time.perf_counter() - t) / 20 * 1e3
    print(f"  lut: scan mean {s.mean():.2f} m | PF 4000x100 {pf:.2f} ms ({25/pf:.1f}x rt)")
    e.save_lut("/tmp/f_lut.npz")
    e2 = RaycastEngine.load_lut("/tmp/f_lut.npz")
    print(f"  load_lut roundtrip scan mean {e2.scan(pose,1080,4.7).mean():.2f} m (no range_libc/numba needed)")

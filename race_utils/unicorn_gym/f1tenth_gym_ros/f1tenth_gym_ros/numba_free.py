"""Run f1tenth_gym WITHOUT numba.

Why this exists
---------------
numba 0.60 hard-rejects numpy >= 2.1 at *import* time, so every
``from numba import njit`` in f110_gym (base_classes / dynamic_models /
collision_models / laser_models) raises before any code runs. Two steps fix it:

1. Install a no-op ``numba`` shim so those imports succeed. The ``@njit``
   functions then run as plain Python — fine for the single-car dynamics /
   collision, but the 1080-beam laser raycast would be unusably slow in pure
   Python, therefore:
2. Replace f110_gym's ``ScanSimulator2D`` with a thin adapter backed by the
   vendored ``range_libc`` **RayMarching ('rm')** C++ backend (verified to match
   the numba sim to ~2.9 cm). The scan stays fast; numba is gone.

Import this module (or call :func:`enable`) BEFORE ``gym.make('f110_gym:...')``.
Override knobs via env vars: ``RAYCASTER_BACKEND`` (default ``rm``) and
``RAYCASTER_DIR`` (path to race_utils/raycaster).
"""
import os
import sys
import types
import importlib.util

import numpy as np

def _find_raycaster_dir():
    # Ascend from this file and look for race_utils/raycaster/raycaster.py under
    # any ancestor. Robust to the source, build/ and install/ layouts (the old
    # fixed dirname() chain pointed at a non-existent path under build/).
    _here = os.path.dirname(os.path.abspath(__file__))
    rels = (
        ("race_utils", "raycaster"),
        ("src", "unicorn-racing-stack", "race_utils", "raycaster"),
        ("unicorn-racing-stack", "race_utils", "raycaster"),
    )
    d = _here
    for _ in range(12):
        for rel in rels:
            cand = os.path.join(d, *rel)
            if os.path.isfile(os.path.join(cand, "raycaster.py")):
                return cand
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.normpath(os.path.join(_here, "..", "..", "..", "race_utils", "raycaster"))


_RAYCASTER_DIR = os.environ.get("RAYCASTER_DIR") or _find_raycaster_dir()


# --------------------------------------------------------------------------- #
# 1) numba shim
# --------------------------------------------------------------------------- #
def _install_numba_shim():
    """Make ``from numba import njit`` a no-op passthrough (unless real numba
    already imports cleanly, in which case leave it alone)."""
    if "numba" in sys.modules:
        return
    try:
        import numba  # noqa: F401  — real numba works here, prefer it
        return
    except Exception:
        pass

    shim = types.ModuleType("numba")

    def njit(*args, **kwargs):
        # @njit (bare)  ->  njit(func)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # @njit(cache=True, ...)  ->  returns the decorator
        def deco(func):
            return func
        return deco

    shim.njit = njit
    shim.jit = njit
    shim.prange = range
    sys.modules["numba"] = shim


# --------------------------------------------------------------------------- #
# 2) range_libc-backed ScanSimulator2D replacement
# --------------------------------------------------------------------------- #
def _load_RaycastEngine():
    path = os.path.join(_RAYCASTER_DIR, "raycaster.py")
    spec = importlib.util.spec_from_file_location("raycaster", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["raycaster"] = mod
    spec.loader.exec_module(mod)
    return mod.RaycastEngine


class RaycastScanSim:
    """Drop-in replacement for f110_gym ``ScanSimulator2D`` (range_libc 'rm')."""

    def __init__(self, num_beams, fov, eps=0.0001, theta_dis=2000, max_range=30.0):
        self.num_beams = int(num_beams)
        self.fov = float(fov)
        self.max_range = float(max_range)
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.map_height = None  # mirrors ScanSimulator2D's "map set?" sentinel

        RaycastEngine = _load_RaycastEngine()
        backend = os.environ.get("RAYCASTER_BACKEND", "rm")
        self.eng = RaycastEngine(backend=backend, max_range_m=self.max_range)

    # --- map (occupancy: True = obstacle, bottom-left origin) ---------------
    def _apply(self, occ, res, ox, oy):
        self.eng.set_map(occ, res, (ox, oy))
        self.map_height, self.map_width = occ.shape
        self.map_resolution = res

    def set_map(self, map_path, map_ext):
        """yaml + image, mirroring ScanSimulator2D.set_map exactly."""
        import yaml
        from PIL import Image
        img_path = os.path.splitext(map_path)[0] + map_ext
        img = np.array(Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        meta = yaml.safe_load(open(map_path, "r"))
        res = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        self._apply(img <= 128.0, res, ox, oy)  # <=128 -> obstacle
        return True

    def set_map_from_array(self, map_img, map_resolution, origin_x, origin_y, origin_theta=0.0):
        """map_img is already bottom-origin (0=obstacle, 255=free)."""
        occ = np.asarray(map_img, np.float64) <= 128.0
        self._apply(occ, float(map_resolution), float(origin_x), float(origin_y))
        return True

    # --- scan ---------------------------------------------------------------
    def scan(self, pose, rng, std_dev=0.01):
        if self.map_height is None:
            raise ValueError("Map is not set for scan simulator.")
        out = self.eng.scan(
            np.asarray(pose, float), self.num_beams, self.fov,
            max_range=self.max_range, miss=None,
        ).astype(np.float64)
        if rng is not None:
            out = out + rng.normal(0.0, std_dev, size=self.num_beams)
        return out

    def get_increment(self):
        return self.angle_increment


def _patch_scan_simulator():
    # importing base_classes pulls dynamic_models/collision_models/laser_models,
    # all of which need the numba shim already in place.
    import f110_gym.envs.base_classes as bc
    bc.ScanSimulator2D = RaycastScanSim

    # PERFORMANCE: gym_bridge overlays the opponent into the PUBLISHED scan with
    # its own range_libc engine, so the gym's internal per-step vehicle->vehicle
    # ray_cast (numba -> pure-Python under the shim) is redundant. Worse, its cost
    # scales with how many beams the opponent covers, so it explodes as the two
    # cars approach -> the "lag when cars get close". No-op it.
    bc.RaceCar.ray_cast_agents = lambda self, scan: scan

    # PERFORMANCE: check_ttc_jit loops over every beam (2160) per agent per step
    # in pure Python under the shim -> baseline lag whenever a car is moving.
    # Replace with a vectorized numpy equivalent (same collision result).
    def _fast_check_ttc(scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
        if vel == 0.0:
            return False
        with np.errstate(divide='ignore', invalid='ignore'):
            ttc = (np.asarray(scan) - side_distances) / (vel * cosines)
        return bool(np.any((ttc < ttc_thresh) & (ttc >= 0.0)))
    bc.check_ttc_jit = _fast_check_ttc


def enable():
    _install_numba_shim()
    _patch_scan_simulator()


# Run on import so a single `import f1tenth_gym_ros.numba_free` is enough.
enable()

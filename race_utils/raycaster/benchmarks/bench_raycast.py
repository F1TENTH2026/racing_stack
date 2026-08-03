#!/usr/bin/env python3
"""
2D LiDAR raycaster benchmark — grid methods + analytic segment method.

  - numba f1tenth_gym      : distance-transform sphere-tracing (CPU, numba JIT)
  - range_libc bl/rm/cddt/pcddt/glt : C++ (Bresenham / RayMarch / CDDT / GiantLUT)
  - segment (analytic)     : ray vs track-boundary line segments (numpy), 2D
                             "z-buffer": nearest segment hit per beam.

Same map / poses / beams for every method. Map via env MAP=test|f (default test).
The segment method only runs when the map dir has boundary_left/right.csv.

Run: MAP=f /tmp/rlbench/bin/python bench_raycast.py
"""
import sys, os, time, importlib.util
import numpy as np
from PIL import Image
import yaml
from scipy.ndimage import distance_transform_edt

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
MAPNAME = os.environ.get("MAP", "test")
MAP_DIR  = f"{CAC}/stack_master/maps/{MAPNAME}"
MAP_YAML = f"{MAP_DIR}/{MAPNAME}.yaml"
GYM_DIR   = CAC + "/simulator/f1tenth_gym/gym"
RL_PYWRAP = CAC + "/slam/range_libc/pywrapper"

NUM_BEAMS, FOV, MAX_RANGE_M = 1080, 4.7, 10.0
N_POSES, REPEAT, THETA_DISC = 500, 3, 112

# ---------- map ----------
meta = yaml.safe_load(open(MAP_YAML))
res = float(meta["resolution"]); origin = meta["origin"]
img = np.array(Image.open(os.path.join(MAP_DIR, meta["image"])))
if img.ndim == 3: img = img[..., 0]
H, W = img.shape
MAX_RANGE_PX = MAX_RANGE_M / res
occupied = (img <= 128); free = ~occupied

clear = distance_transform_edt(free)
ys, xs = np.where(clear > 5)
rngp = np.random.default_rng(0)
sel = rngp.choice(len(xs), size=min(N_POSES, len(xs)), replace=False)
col = xs[sel].astype(np.float32); row = ys[sel].astype(np.float32)
th  = rngp.uniform(-np.pi, np.pi, len(sel)).astype(np.float32)
N = len(sel)
wx = (origin[0] + col * res).astype(np.float64)
wy = (origin[1] + (H - 1 - row) * res).astype(np.float64)
wth = th.astype(np.float64)
beam_off = np.linspace(-FOV / 2, FOV / 2, NUM_BEAMS).astype(np.float64)

print(f"MAP={MAPNAME} {W}x{H}px @ {res} m/px | beams={NUM_BEAMS} fov={FOV} "
      f"max_range={MAX_RANGE_M}m | poses={N} x{REPEAT} | occupied={occupied.mean()*100:.1f}%")
results = []   # (name, ms, init_ms, mean_m)

# ---------- numba f1tenth_gym ----------
spec = importlib.util.spec_from_file_location(
    "laser_models", os.path.join(GYM_DIR, "f110_gym/envs/laser_models.py"))
lm = importlib.util.module_from_spec(spec); sys.modules["laser_models"] = lm
spec.loader.exec_module(lm)
sim = lm.ScanSimulator2D(NUM_BEAMS, FOV, max_range=MAX_RANGE_M)
t = time.perf_counter(); sim.set_map(MAP_YAML, "." + meta["image"].split(".")[-1])
init = (time.perf_counter() - t) * 1e3
sim.scan(np.array([wx[0], wy[0], wth[0]]), None)
sample = sim.scan(np.array([wx[0], wy[0], wth[0]]), None)
t0 = time.perf_counter()
for _ in range(REPEAT):
    for i in range(N): sim.scan(np.array([wx[i], wy[i], wth[i]]), None)
ms = (time.perf_counter() - t0) / (REPEAT * N) * 1e3
results.append(("numba f1tenth_gym (CPU)", ms, init, float(np.mean(sample))))

# ---------- range_libc ----------
sys.path.insert(0, RL_PYWRAP)
import range_libc
oMap = range_libc.PyOMap(occupied)
angles = beam_off.astype(np.float32)
def bench_rl(name, ctor):
    t = time.perf_counter(); m = ctor(); init = (time.perf_counter() - t) * 1e3
    q = np.zeros((NUM_BEAMS, 3), np.float32); out = np.zeros(NUM_BEAMS, np.float32)
    q[:, 0] = col[0]; q[:, 1] = row[0]; q[:, 2] = th[0] + angles; m.calc_range_many(q, out)
    smpl = float(np.mean(out)) * res
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        for i in range(N):
            q[:, 0] = col[i]; q[:, 1] = row[i]; q[:, 2] = th[i] + angles
            m.calc_range_many(q, out)
    ms = (time.perf_counter() - t0) / (REPEAT * N) * 1e3
    results.append((name, ms, init, smpl))
bench_rl("range_libc bl  (Bresenham)",  lambda: range_libc.PyBresenhamsLine(oMap, MAX_RANGE_PX))
bench_rl("range_libc rm  (RayMarching)", lambda: range_libc.PyRayMarching(oMap, MAX_RANGE_PX))
bench_rl("range_libc cddt",              lambda: range_libc.PyCDDTCast(oMap, MAX_RANGE_PX, THETA_DISC))
bench_rl("range_libc pcddt (PF default)", lambda: (lambda m: (m.prune(), m)[1])(range_libc.PyCDDTCast(oMap, MAX_RANGE_PX, THETA_DISC)))
bench_rl("range_libc glt (GiantLUT)",    lambda: range_libc.PyGiantLUTCast(oMap, MAX_RANGE_PX, THETA_DISC))

# ---------- analytic segment raycaster (track boundaries) ----------
bl = f"{MAP_DIR}/boundary_left.csv"; br = f"{MAP_DIR}/boundary_right.csv"
if os.path.exists(bl) and os.path.exists(br):
    def load_loop(path):
        p = np.loadtxt(path, delimiter=",", skiprows=1)[:, :2]
        return p
    pts = [load_loop(bl), load_loop(br)]
    A = []; B = []
    for p in pts:                                  # consecutive pts -> segments, close loop
        a = p; b = np.roll(p, -1, axis=0)
        A.append(a); B.append(b)
    A = np.concatenate(A).astype(np.float64); B = np.concatenate(B).astype(np.float64)
    E = B - A                                       # segment dir [S,2]
    S = len(A)
    t = time.perf_counter()                         # "init" = nothing precomputed
    init = (time.perf_counter() - t) * 1e3
    def seg_scan(px, py, theta):
        ang = theta + beam_off
        dx = np.cos(ang); dy = np.sin(ang)          # [Bm]
        wx_ = A[:, 0] - px; wy_ = A[:, 1] - py       # [S]
        det = E[:, 0][None, :] * dy[:, None] - E[:, 1][None, :] * dx[:, None]   # [Bm,S]
        # t_ray = (E x w)/det ; u = (d x w)/det
        tr = (E[:, 0][None, :] * wy_[None, :] - E[:, 1][None, :] * wx_[None, :]) / det
        u  = (dx[:, None] * wy_[None, :] - dy[:, None] * wx_[None, :]) / det
        ok = (det != 0) & (tr >= 0) & (u >= 0) & (u <= 1) & (tr <= MAX_RANGE_M)
        tr = np.where(ok, tr, np.inf)
        return np.minimum(tr.min(axis=1), MAX_RANGE_M)
    smpl = float(np.mean(seg_scan(wx[0], wy[0], wth[0])))
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        for i in range(N): seg_scan(wx[i], wy[i], wth[i])
    ms = (time.perf_counter() - t0) / (REPEAT * N) * 1e3
    results.append((f"segment analytic ({S} segs, CPU)", ms, init, smpl))

# ---------- report ----------
base = results[0][1]
print(f"\n{'method':<34}{'init ms':>9}{'ms/scan':>10}{'scans/s':>10}{'vs numba':>10}{'mean m':>8}")
print("-" * 81)
for name, ms, init, mr in results:
    print(f"{name:<34}{init:>9.1f}{ms:>10.4f}{1000/ms:>10.0f}{base/ms:>9.1f}x{mr:>8.2f}")

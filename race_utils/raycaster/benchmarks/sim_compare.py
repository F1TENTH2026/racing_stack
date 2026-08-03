#!/usr/bin/env python3
"""
Simulator drop-in comparison.

Existing simulator raycaster = f1tenth_gym numba ScanSimulator2D.
New = RaycastEngine (vendored range_libc), backends pcddt + lut.

Same map, same world poses, same 1080 beams. Reports (1) output agreement
(can RaycastEngine replace the sim's raycaster?) and (2) per-scan speed.

Run: NUMBA_CACHE_DIR=/tmp/nc /tmp/rlbench/bin/python sim_compare.py
"""
import os, sys, time, importlib.util
import numpy as np
import yaml as _yaml
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/raycaster
sys.path.insert(0, HERE)
from raycaster import RaycastEngine

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
MAPNAME = os.environ.get("MAP", "f")
YAML = f"{CAC}/stack_master/maps/{MAPNAME}/{MAPNAME}.yaml"
NUM_BEAMS, FOV, MAXR, N = 1080, 4.7, 10.0, 300

# ---- existing sim raycaster (numba ScanSimulator2D) ----
spec = importlib.util.spec_from_file_location(
    "laser_models", CAC + "/simulator/f1tenth_gym/gym/f110_gym/envs/laser_models.py")
lm = importlib.util.module_from_spec(spec); sys.modules["laser_models"] = lm; spec.loader.exec_module(lm)
sim = lm.ScanSimulator2D(NUM_BEAMS, FOV, max_range=MAXR)
ext = "." + _yaml.safe_load(open(YAML))["image"].split(".")[-1]
sim.set_map(YAML, ext)

# ---- new unified engine (vendored range_libc) ----
occ, res, origin = RaycastEngine.load_map_yaml(YAML)
TD = int(os.environ.get("TD", "720"))
print(f"building LUT backend (numba oracle, theta_disc={TD})...")
engines = {f"lut(TD={TD})": RaycastEngine(backend="lut", max_range_m=MAXR, theta_disc=TD).set_map(occ, res, origin)}

# ---- free world poses inside the track ----
clear = distance_transform_edt(~occ); ys, xs = np.where(clear > 4)
rngp = np.random.default_rng(1); sel = rngp.choice(len(xs), N, replace=False)
poses = np.stack([origin[0] + xs[sel]*res, origin[1] + ys[sel]*res,
                  rngp.uniform(-np.pi, np.pi, N)], 1)

print(f"MAP={MAPNAME} | {NUM_BEAMS} beams, fov={FOV}, max={MAXR} m | {N} poses")
print("existing = numba ScanSimulator2D (current simulator raycaster)\n")

sim.scan(poses[0], None)                                   # numba warmup
base = np.array([sim.scan(p, None) for p in poses])        # [N, beams]
for b, e in engines.items():
    new = np.array([e.scan(p, NUM_BEAMS, FOV) for p in poses])
    d = np.abs(new - base)
    mask = ~((base >= MAXR - 1e-3) & (new >= MAXR - 1e-3))  # ignore both-saturated beams
    print(f"  {b:6s} vs existing : MAE {d[mask].mean()*100:5.1f} cm | "
          f"p95 {np.percentile(d[mask],95)*100:5.1f} cm | within 10cm {(d<0.10).mean()*100:5.1f}%")

def t_scan(fn):
    fn(poses[0]); t0 = time.perf_counter()
    for _ in range(3):
        for p in poses: fn(p)
    return (time.perf_counter() - t0) / (3 * N) * 1e3
ms_sim = t_scan(lambda p: sim.scan(p, None))
print(f"\n  existing numba   : {ms_sim:7.4f} ms/scan  ({1000/ms_sim:6.0f}/s)")
for b, e in engines.items():
    ms = t_scan(lambda p, e=e: e.scan(p, NUM_BEAMS, FOV))
    print(f"  RaycastEngine {b:4s}: {ms:7.4f} ms/scan  ({1000/ms:6.0f}/s)  {ms_sim/ms:.1f}x vs numba")

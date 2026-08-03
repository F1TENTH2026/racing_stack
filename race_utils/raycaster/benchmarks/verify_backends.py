#!/usr/bin/env python3
"""
Verify ALL RaycastEngine backends against the numba f1tenth_gym sim — correctness
(MAE per beam) and speed — for both the simulator scan() and the particle-filter
calc_range_repeat_angles() paths. Uses the FIXED range_libc (world-coord queries).

Run: NUMBA_CACHE_DIR=/tmp/nc MAP=f /tmp/jz/bin/python verify_backends.py
"""
import os, sys, importlib.util, time
import numpy as np
import yaml as _yaml
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from raycaster import RaycastEngine

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
MAP = os.environ.get("MAP", "f")
YAML = f"{CAC}/stack_master/maps/{MAP}/{MAP}.yaml"
NB, FOV, MR, N = 1080, 4.7, 10.0, 300

spec = importlib.util.spec_from_file_location(
    "lm", CAC + "/simulator/f1tenth_gym/gym/f110_gym/envs/laser_models.py")
lm = importlib.util.module_from_spec(spec); sys.modules["lm"] = lm; spec.loader.exec_module(lm)
sim = lm.ScanSimulator2D(NB, FOV, max_range=MR)
sim.set_map(YAML, "." + _yaml.safe_load(open(YAML))["image"].split(".")[-1])
occ, res, origin = RaycastEngine.load_map_yaml(YAML)

clear = distance_transform_edt(~occ); ys, xs = np.where(clear > 6)
H, W = occ.shape; mm = (xs > 20) & (xs < W - 20) & (ys > 20) & (ys < H - 20)
xs, ys = xs[mm], ys[mm]; r = np.random.default_rng(7); s = r.choice(len(xs), N, False)
poses = np.stack([origin[0] + xs[s] * res, origin[1] + ys[s] * res, r.uniform(-np.pi, np.pi, N)], 1)
sim.scan(poses[0], None); base = np.array([sim.scan(p, None) for p in poses])

def t_scan(fn):
    fn(poses[0]); t0 = time.perf_counter()
    for _ in range(3):
        for p in poses: fn(p)
    return (time.perf_counter() - t0) / (3 * N) * 1e3

print(f"MAP={MAP} {occ.shape} | {NB} beams fov={FOV} max={MR}m | {N} poses")
print("existing = numba ScanSimulator2D\n")
print(f"{'backend':<16}{'sim MAE':>9}{'within10cm':>11}{'ms/scan':>9}{'vs numba':>9}")
print("-" * 56)
ms_n = t_scan(lambda p: sim.scan(p, None))
print(f"{'numba (ref)':<16}{'-':>9}{'-':>11}{ms_n:>9.4f}{'1.0x':>9}")
for be in ["rm", "cddt", "pcddt", "glt", "bl", "lut"]:
    td = 720 if be == "lut" else 112
    e = RaycastEngine(backend=be, max_range_m=MR, theta_disc=td).set_map(occ, res, origin)
    new = np.array([e.scan(p, NB, FOV) for p in poses])
    d = np.abs(new - base); mask = ~((base >= MR - 1e-3) & (new >= MR - 1e-3))
    ms = t_scan(lambda p, e=e: e.scan(p, NB, FOV))
    print(f"{be:<16}{d[mask].mean()*100:>8.1f}c{(d<0.1).mean()*100:>10.1f}%{ms:>9.4f}{ms_n/ms:>8.1f}x")

# ---- PF path: calc_range_repeat_angles correctness (pcddt) vs numba scan ----
print("\nPF path (calc_range_repeat_angles) vs numba, 60 downsampled beams:")
K = 60; ang = np.linspace(-FOV/2, FOV/2, K).astype(np.float32)
sim_pf = lm.ScanSimulator2D(K, FOV, max_range=MR); sim_pf.set_map(YAML, "." + _yaml.safe_load(open(YAML))["image"].split(".")[-1])
base_pf = np.array([sim_pf.scan(p, None) for p in poses[:50]])
for be in ["pcddt", "glt", "lut"]:
    e = RaycastEngine(backend=be, max_range_m=MR, theta_disc=(720 if be == "lut" else 112)).set_map(occ, res, origin)
    rr = e.calc_range_repeat_angles(poses[:50].astype(np.float32), ang).reshape(50, K)
    d = np.abs(rr - base_pf); mask = ~((base_pf >= MR-1e-3) & (rr >= MR-1e-3))
    print(f"  {be:<8}: MAE {d[mask].mean()*100:.1f} cm | within 10cm {(d<0.1).mean()*100:.1f}%")

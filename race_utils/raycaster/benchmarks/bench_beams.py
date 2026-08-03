#!/usr/bin/env python3
"""
Real-time feasibility at realistic beam counts: 1080 and 2160 beams @ 40 Hz.
Budget = 25 ms/scan (40 Hz). Per backend, per beam count.

Run (system python with range_libc built for it; numba optional for lut):
  NUMBA_CACHE_DIR=/tmp/nc /usr/bin/python3 bench_beams.py
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from raycaster import RaycastEngine

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
YAML = f"{CAC}/stack_master/maps/f/f.yaml"
MR, BUDGET = 10.0, 25.0   # 40 Hz
occ, res, origin = RaycastEngine.load_map_yaml(YAML)
ys, xs = np.where(~occ); i = len(xs) // 2
pose = np.array([origin[0] + xs[i] * res, origin[1] + ys[i] * res, 0.3])

backends = ["rm", "glt", "pcddt"]
try:
    import numba  # noqa
    backends.append("lut")
except Exception:
    pass

print(f"40 Hz budget = {BUDGET:.0f} ms/scan | map f | backends: {backends}\n")
print(f"{'backend':<10}{'1080 ms':>9}{'40Hz':>6}{'2160 ms':>10}{'40Hz':>6}{'init':>8}")
print("-" * 50)
for be in backends:
    t = time.perf_counter()
    e = RaycastEngine(be, max_range_m=MR, theta_disc=(720 if be == "lut" else 112)).set_map(occ, res, origin)
    init = (time.perf_counter() - t) * 1e3
    row = []
    for nb in (1080, 2160):
        e.scan(pose, nb, 4.7)  # warmup
        t = time.perf_counter()
        for _ in range(200): e.scan(pose, nb, 4.7)
        ms = (time.perf_counter() - t) / 200 * 1e3
        row.append((ms, "OK" if ms < BUDGET else "FAIL"))
    print(f"{be:<10}{row[0][0]:>9.3f}{row[0][1]:>6}{row[1][0]:>10.3f}{row[1][1]:>6}{init:>7.0f}ms")

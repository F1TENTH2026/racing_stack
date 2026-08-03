#!/usr/bin/env python3
"""Quickstart — load the bundled sample map and cast one LiDAR scan.

    python examples/quickstart.py

Uses maps/sample_map.{png,yaml} shipped in this repo. range_libc must be built
once (`pip install -e range_libc/pywrapper`); raycaster.py is imported from the
repo root.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                                   # raycaster.py
sys.path.insert(0, os.path.join(ROOT, "range_libc", "pywrapper"))   # range_libc (if not pip-installed)

from raycaster import RaycastEngine

MAP = os.path.join(ROOT, "maps", "sample_map.yaml")

# ---- high level: RaycastEngine (what the sim + particle filter share) -------
occ, res, origin = RaycastEngine.load_map_yaml(MAP)        # occ[H,W] bool, row0 = bottom
print(f"map: {occ.shape[1]}x{occ.shape[0]} px @ {res} m/px  origin {origin}")

eng = RaycastEngine(backend="rm", max_range_m=10.0, theta_disc=360).set_map(occ, res, origin)
pose = [7.5, 2.5, 0.0]                                     # x, y (m), heading (rad) — a free spot
ranges = eng.scan(pose, 1080, 4.7)                        # one 1080-beam, 270° scan
print(f"scan : {len(ranges)} beams | min {ranges.min():.2f}  mean {ranges.mean():.2f}  max {ranges.max():.2f} m")

# ---- low level: range_libc directly (same map, same result) -----------------
import range_libc
omap = range_libc.PyOMap(occ, resolution=res, origin_x=origin[0], origin_y=origin[1])
rm = range_libc.PyRayMarching(omap, 10.0 / res)
n, fov = 1080, 4.7
q = np.zeros((n, 3), np.float32)
q[:, 0], q[:, 1] = pose[0], pose[1]
q[:, 2] = pose[2] + np.linspace(-fov / 2, fov / 2, n).astype(np.float32)
out = np.zeros(n, np.float32)
rm.calc_range_many(q, out)                                # queries[N,3] in world meters
print(f"range_libc: min {out.min():.2f}  mean {out.mean():.2f}  max {out.max():.2f} m")

#!/usr/bin/env python3
"""
Verify dynamic overlay on a PRECOMPUTED static raycaster (no map rebuild):
  - static scan from glt/lut (table built ONCE)
  - obstacle (circle) and opponent (box) appear via min(static, ray) overlay
  - rate structure: lidar 40 Hz, dynamics 120 Hz, opponent traced as it moves

Run: NUMBA_CACHE_DIR=/tmp/nc /usr/bin/python3 benchmarks/verify_overlay.py
"""
import os, sys, time
import numpy as np
from scipy.ndimage import distance_transform_edt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from raycaster import RaycastEngine

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
occ, res, origin = RaycastEngine.load_map_yaml(f"{CAC}/stack_master/maps/f/f.yaml")
NB, FOV, MR = 2160, 4.7, 10.0
yc, xc = np.unravel_index(np.argmax(distance_transform_edt(~occ)), occ.shape)   # most open spot
pose = np.array([origin[0] + xc * res, origin[1] + yc * res, 0.0])
ang = pose[2] + np.linspace(-FOV / 2, FOV / 2, NB)

print(f"pose {pose.round(2)} | {NB} beams\n")
print("=== overlay correctness (static table built once, dynamics via overlay) ===")
for be in ["glt", "lut"]:
    e = RaycastEngine(be, MR, theta_disc=(720 if be == "lut" else 112)).set_map(occ, res, origin)
    base = e.scan(pose, NB, FOV)
    bi = int(np.argmax(base))                                   # an open beam
    a = ang[bi]; dobs = float(base[bi]) * 0.4
    obs = [[pose[0] + dobs * np.cos(a), pose[1] + dobs * np.sin(a), 0.25]]   # r=0.25 m
    s_obs = e.scan_with_dynamics(pose, NB, FOV, obstacles=obs)
    bj = int(np.argmax(np.where(np.abs(np.arange(NB) - bi) > 200, base, 0)))  # another open beam
    aj = ang[bj]; dopp = float(base[bj]) * 0.5
    opp = [[pose[0] + dopp * np.cos(aj), pose[1] + dopp * np.sin(aj), aj + np.pi / 2]]
    s_opp = e.scan_with_dynamics(pose, NB, FOV, opp_poses=opp)
    print(f"  {be:4s}: obstacle beam {base[bi]:5.2f}->{s_obs[bi]:4.2f} m (placed {dobs:.2f}) "
          f"| opponent beam {base[bj]:5.2f}->{s_opp[bj]:4.2f} m (placed {dopp:.2f})")
    t = time.perf_counter()
    for _ in range(300): e.scan_with_dynamics(pose, NB, FOV, obstacles=obs, opp_poses=opp)
    print(f"        static+overlay {NB} beams: {(time.perf_counter()-t)/300*1e3:.3f} ms/scan "
          f"(40 Hz budget 25 ms)")

print("\n=== rate structure: lidar 40 Hz, dynamics 120 Hz, moving opponent (glt) ===")
e = RaycastEngine("glt", MR, theta_disc=112).set_map(occ, res, origin)
base = e.scan(pose, NB, FOV)
bo = int(np.argmax(base)); ao = ang[bo]                         # an OPEN beam (range = max)
dvec = np.array([np.cos(ao), np.sin(ao)]); pvec = np.array([-np.sin(ao), np.sin(np.pi/2 + ao - ao) * 0 + np.cos(ao)])
R = 4.0                                                         # opponent crosses at 4 m along the open beam
dt_dyn = 1.0 / 120; lidar_every = 3                            # 120 / 40
s_lat = -2.0; vlat = 1.4                                        # opponent slides across the beam
samples = 0; t0 = time.perf_counter()
print(f"  open beam #{bo} static range = {base[bo]:.2f} m (max); opponent crosses it at {R:.0f} m")
for k in range(int(3.0 / dt_dyn)):
    s_lat += vlat * dt_dyn                                      # dynamics @ 120 Hz
    if k % lidar_every == 0:                                    # lidar @ 40 Hz
        opp_c = pose[:2] + R * dvec + s_lat * pvec
        s = e.scan_with_dynamics(pose, NB, FOV, opp_poses=[[opp_c[0], opp_c[1], ao]]); samples += 1
        if k % 24 == 0:
            tag = "  <- opponent in beam" if s[bo] < base[bo] - 0.5 else ""
            print(f"   t={k*dt_dyn:4.2f}s  lateral={s_lat:+5.2f} m  open_beam={s[bo]:5.2f} m{tag}")
dur = time.perf_counter() - t0
print(f"  ran {samples} lidar frames @ 40 Hz with 120 Hz dynamics in {dur:.3f}s — "
      f"static glt table built ONCE, never rebuilt")

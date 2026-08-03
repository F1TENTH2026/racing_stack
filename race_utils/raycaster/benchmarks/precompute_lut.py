#!/usr/bin/env python3
"""
Precompute → save → load test for the Giant-LUT raycaster.

range_libc exposes NO save/load for its GLT/CDDT tables. So we materialize the
full (x, y, theta) → range table once (using range_libc GLT as the oracle),
save it as a compressed .npz, and at runtime memory-map it and answer scans with
pure-numpy fancy indexing — i.e. NO range_libc needed at query time (portable to
NUC / Mac mini where building range_libc is a hassle).

Tests on the `f` map: build time, materialize time, file size, load time,
query throughput (PF batch + sim), and correctness vs range_libc GLT.

Run: MAP=f /tmp/rlbench/bin/python precompute_lut.py
"""
import os, sys, time
import numpy as np
from PIL import Image
import yaml

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
RL = CAC + "/slam/range_libc/pywrapper"
MAPNAME = os.environ.get("MAP", "f")
MAP_DIR = f"{CAC}/stack_master/maps/{MAPNAME}"; MAP_YAML = f"{MAP_DIR}/{MAPNAME}.yaml"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "precomputed")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RANGE_M, THETA_DISC = 10.0, 112

# ---- map ----
meta = yaml.safe_load(open(MAP_YAML)); res = float(meta["resolution"]); origin = meta["origin"]
img = np.array(Image.open(os.path.join(MAP_DIR, meta["image"])))
if img.ndim == 3: img = img[..., 0]
occupied = (img <= 128); H, W = occupied.shape
MAX_RANGE_PX = MAX_RANGE_M / res
dtheta = 2 * np.pi / THETA_DISC

sys.path.insert(0, RL); import range_libc
print(f"MAP={MAPNAME} {W}x{H} @ {res} m/px | theta_disc={THETA_DISC} | max_range={MAX_RANGE_M} m")

# ================= 1) build GLT (range_libc) =================
t = time.perf_counter(); oMap = range_libc.PyOMap(occupied)
glt = range_libc.PyGiantLUTCast(oMap, MAX_RANGE_PX, THETA_DISC)
build_s = time.perf_counter() - t
print(f"[build]  range_libc GLT construct: {build_s*1e3:.0f} ms")

# ================= 2) materialize full LUT -> numpy =================
# LUT[x, y, k] = range_px at angle k*dtheta. Filled per-theta with calc_range_many.
t = time.perf_counter()
lut = np.zeros((W, H, THETA_DISC), dtype=np.uint16)
xs = np.repeat(np.arange(W, dtype=np.float32), H)
ys = np.tile(np.arange(H, dtype=np.float32), W)
q = np.zeros((W * H, 3), dtype=np.float32); q[:, 0] = xs; q[:, 1] = ys
out = np.zeros(W * H, dtype=np.float32)
for k in range(THETA_DISC):
    q[:, 2] = k * dtheta
    glt.calc_range_many(q, out)
    lut[:, :, k] = np.minimum(out, MAX_RANGE_PX).reshape(W, H).astype(np.uint16)
mat_s = time.perf_counter() - t
print(f"[materialize] {W*H*THETA_DISC:,} entries via calc_range_many: {mat_s:.2f} s")

# ================= 3) save =================
path = os.path.join(OUT_DIR, f"{MAPNAME}_glt_td{THETA_DISC}_mr{int(MAX_RANGE_M)}.npz")
t = time.perf_counter()
np.savez_compressed(path, lut=lut, resolution=res, origin=np.array(origin, np.float64),
                    max_range_m=MAX_RANGE_M, theta_disc=THETA_DISC, W=W, H=H)
save_s = time.perf_counter() - t
mb = os.path.getsize(path) / 1024**2
print(f"[save]   {os.path.basename(path)}  {mb:.1f} MB  ({save_s:.2f} s)")

# ================= 4) load (mmap) =================
t = time.perf_counter()
z = np.load(path)
lut2 = z["lut"]; res2 = float(z["resolution"]); td2 = int(z["theta_disc"])
load_s = time.perf_counter() - t
print(f"[load]   {load_s*1e3:.0f} ms  (shape {lut2.shape}, {lut2.dtype})")

# ================= 5) pure-numpy LUT scan (no range_libc) =================
def lut_scan_batch(px, py, theta, beam_off):
    """px,py [M] pixel, theta [M], beam_off [K] -> ranges_m [M,K] via numpy gather."""
    xi = np.clip(px.astype(np.int64), 0, W - 1); yi = np.clip(py.astype(np.int64), 0, H - 1)
    ang = theta[:, None] + beam_off[None, :]                       # [M,K]
    ki = np.mod(np.rint(ang / dtheta).astype(np.int64), td2)        # [M,K]
    return lut2[xi[:, None], yi[:, None], ki] * res2               # [M,K] meters

# PF workload: 4000 particles x 100 beams
rngp = np.random.default_rng(0)
clear = (~occupied)
ys_, xs_ = np.where(clear)
sel = rngp.choice(len(xs_), 4000, replace=True)
px = xs_[sel].astype(np.float32); py = ys_[sel].astype(np.float32)
th = rngp.uniform(-np.pi, np.pi, 4000).astype(np.float32)
boff = np.linspace(-4.7/2, 4.7/2, 100).astype(np.float32)
lut_scan_batch(px[:1], py[:1], th[:1], boff)                        # warmup
t = time.perf_counter()
for _ in range(20): r = lut_scan_batch(px, py, th, boff)
pf_ms = (time.perf_counter() - t) / 20 * 1e3
print(f"\n[query]  numpy-LUT PF batch (4000x100): {pf_ms:.3f} ms/update  ->  {25/pf_ms:.1f}x real-time")

# sim: 1 pose x 2200 beams
boff_s = np.linspace(-4.7/2, 4.7/2, 2200).astype(np.float32)
t = time.perf_counter()
for _ in range(200): lut_scan_batch(px[:1], py[:1], th[:1], boff_s)
sim_ms = (time.perf_counter() - t) / 200 * 1e3
print(f"[query]  numpy-LUT sim (1x2200):        {sim_ms:.4f} ms/scan")

# ================= 6) correctness vs range_libc GLT =================
qC = np.zeros((100, 3), np.float32); outC = np.zeros(100, np.float32)
err = []
for m in range(50):
    qC[:, 0] = px[m]; qC[:, 1] = py[m]; qC[:, 2] = th[m] + boff
    glt.calc_range_many(qC, outC)
    lut_r = lut_scan_batch(px[m:m+1], py[m:m+1], th[m:m+1], boff)[0] / res2  # px
    err.append(np.abs(outC - lut_r).mean() * res2)
print(f"\n[verify] numpy-LUT vs range_libc GLT: mean abs err = {np.mean(err)*100:.2f} cm "
      f"(angle quantization to {THETA_DISC} bins)")

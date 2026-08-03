#!/usr/bin/env python3
"""
JAX GPU raycaster benchmark — faithful port of the f1tenth_gym / jax_pf
distance-transform sphere-tracing algorithm (same math as numba get_scan),
vmapped over beams and poses, jit-compiled, run on the RTX 4080.

Reports BOTH:
  - single-pose latency  (one scan per dispatch — what a 1-2 agent sim sees)
  - batched throughput    (all poses in one vmap — jax's real strength, e.g.
                           many parallel envs / particle filter)

Same map / poses / beams as bench_raycast.py.
Run: /tmp/rlbench/bin/python bench_jax.py
"""
import os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from PIL import Image
import yaml
from scipy.ndimage import distance_transform_edt as edt
import jax, jax.numpy as jnp
from functools import partial

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
MAPNAME = os.environ.get("MAP", "test")
MAP_DIR = CAC + f"/stack_master/maps/{MAPNAME}"
MAP_YAML = MAP_DIR + f"/{MAPNAME}.yaml"
NUM_BEAMS, FOV, MAX_RANGE_M = 1080, 4.7, 10.0
N_POSES, REPEAT, EPS, MAX_ITERS = 500, 3, 1e-4, 128

print("jax", jax.__version__, "device:", jax.devices()[0])

# ---- map (same convention as f1tenth_gym: flip top-bottom, dt in meters) ----
meta = yaml.safe_load(open(MAP_YAML))
res = float(meta["resolution"]); ox, oy, _ = meta["origin"]
img = np.array(Image.open(os.path.join(MAP_DIR, meta["image"])))
if img.ndim == 3: img = img[..., 0]
mflip = np.flipud(img)
free = (mflip > 128).astype(np.float32)          # 1=free, 0=obstacle
dt_np = (edt(free) * res).astype(np.float32)      # meters to nearest obstacle
H, W = dt_np.shape
dt = jnp.asarray(dt_np)

# ---- same poses as bench_raycast.py (same seed) ----
occupied = (mflip <= 128)
clear = edt(~occupied)
ys, xs = np.where(clear > 5)
rngp = np.random.default_rng(0)
sel = rngp.choice(len(xs), size=min(N_POSES, len(xs)), replace=False)
col = xs[sel].astype(np.float32); row = ys[sel].astype(np.float32)
th = rngp.uniform(-np.pi, np.pi, len(sel)).astype(np.float32)
N = len(sel)
wx = (ox + col * res); wy = (oy + row * res)       # note: row already in flipped frame
poses = jnp.asarray(np.stack([wx, wy, th], axis=1).astype(np.float32))  # (N,3)
beam_off = jnp.asarray(np.linspace(-FOV / 2, FOV / 2, NUM_BEAMS).astype(np.float32))

# ---- jax sphere-tracing raycaster ----
def lookup(x, y):
    c = jnp.clip((( x - ox) / res).astype(jnp.int32), 0, W - 1)
    r = jnp.clip((( y - oy) / res).astype(jnp.int32), 0, H - 1)
    return dt[r, c]

def trace(px, py, ang):
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    d0 = lookup(px, py)
    def cond(st):
        d, total, x, y, it = st
        return (d > EPS) & (total <= MAX_RANGE_M) & (it < MAX_ITERS)
    def body(st):
        d, total, x, y, it = st
        x2, y2 = x + d * cs, y + d * sn
        d2 = lookup(x2, y2)
        return (d2, total + d2, x2, y2, it + 1)
    d, total, *_ = jax.lax.while_loop(cond, body, (d0, d0, px, py, 0))
    return jnp.minimum(total, MAX_RANGE_M)

def scan_one(pose):                                # (3,) -> (NUM_BEAMS,)
    angs = pose[2] + beam_off
    return jax.vmap(lambda a: trace(pose[0], pose[1], a))(angs)

scan_one_jit = jax.jit(scan_one)
scan_batch_jit = jax.jit(jax.vmap(scan_one))       # (N,3) -> (N,NUM_BEAMS)

# ---- warmup (compile) ----
s = scan_one_jit(poses[0]); s.block_until_ready()
sb = scan_batch_jit(poses); sb.block_until_ready()
mean_m = float(np.asarray(sb).mean())

# ---- single-pose latency ----
t0 = time.perf_counter()
for _ in range(REPEAT):
    for i in range(N):
        scan_one_jit(poses[i]).block_until_ready()
single_ms = (time.perf_counter() - t0) / (REPEAT * N) * 1e3

# ---- batched throughput ----
t0 = time.perf_counter()
for _ in range(REPEAT):
    scan_batch_jit(poses).block_until_ready()
batch_total = (time.perf_counter() - t0) / REPEAT
batch_ms = batch_total / N * 1e3

print()
print(f"{'jax mode':<34}{'ms/scan':>10}{'scans/s':>11}{'mean m':>8}")
print("-" * 63)
print(f"{'jax single-pose (GPU, 1 dispatch)':<34}{single_ms:>10.4f}{1000/single_ms:>11.0f}{mean_m:>8.2f}")
print(f"{'jax batched '+str(N)+' poses (GPU)':<34}{batch_ms:>10.4f}{1000/batch_ms:>11.0f}{mean_m:>8.2f}")

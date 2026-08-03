#!/usr/bin/env python3
"""
jax raycaster benchmark — same map + same two workloads as run_bench.py, so the
numbers drop straight into the results table as the `jax-rm(...)` rows.

jax is a distance-transform sphere-tracer (the SAME algorithm as range_libc `rm`
and numba) running on XLA; its value is GPU + large batch (the PF workload).

Run it in its OWN venv so it doesn't disturb the ROS numpy:

    python -m venv /tmp/jaxbench
    /tmp/jaxbench/bin/pip install -U "jax[cuda12]" numpy pillow pyyaml scipy
    /tmp/jaxbench/bin/python bench_jax.py                     # GPU
    JAX_PLATFORMS=cpu /tmp/jaxbench/bin/python bench_jax.py   # CPU

Prints a row matching run_bench.py's format (SIM = 1 scan, PF = 4000x100).
"""
import os
import time
import platform
import numpy as np
from PIL import Image
import yaml
from scipy.ndimage import distance_transform_edt
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

FOV, MAXR = 4.7, 10.0
SIM_BEAMS = 1080
PF_PARTICLES, PF_BEAMS = 4000, 100
HZ = 40
BUDGET_MS = 1000.0 / HZ
MAPNAME = os.environ.get("MAP", "test")


def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


map_dir = os.path.join(REPO, "stack_master", "maps", MAPNAME)
meta = yaml.safe_load(open(os.path.join(map_dir, f"{MAPNAME}.yaml")))
res = float(meta["resolution"])
origin = meta["origin"]
img = np.array(Image.open(os.path.join(map_dir, meta["image"])))
if img.ndim == 3:
    img = img[..., 0]
H, W = img.shape

# distance transform in METERS, y-flipped to world (ROS bottom-origin)
mflip = np.flipud(img)
free = (mflip > 128).astype(np.float32)
dt = jnp.asarray((distance_transform_edt(free) * res).astype(np.float32))
ox, oy = origin[0], origin[1]

# sample free poses (world meters)
occ = (img <= 128)
clear = distance_transform_edt(~occ)
ys, xs = np.where(clear > 5)
rng = np.random.default_rng(0)
sel = rng.choice(len(xs), size=PF_PARTICLES, replace=True)
wx = (origin[0] + xs[sel] * res).astype(np.float32)
wy = (origin[1] + (H - 1 - ys[sel]) * res).astype(np.float32)
wth = rng.uniform(-np.pi, np.pi, PF_PARTICLES).astype(np.float32)


def lookup(x, y):
    c = jnp.clip(((x - ox) / res).astype(jnp.int32), 0, W - 1)
    r = jnp.clip(((y - oy) / res).astype(jnp.int32), 0, H - 1)
    return dt[r, c]


def trace(px, py, ang):
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    d0 = lookup(px, py)

    def cond(st):
        d, tot, x, y, it = st
        return (d > 1e-4) & (tot <= MAXR) & (it < 128)

    def body(st):
        d, tot, x, y, it = st
        x2 = x + d * cs; y2 = y + d * sn
        d2 = lookup(x2, y2)
        return (d2, tot + d2, x2, y2, it + 1)

    d, tot, *_ = jax.lax.while_loop(cond, body, (d0, d0, px, py, 0))
    return jnp.minimum(tot, MAXR)


def make_scan(nbeams):
    off = jnp.asarray(np.linspace(-FOV / 2, FOV / 2, nbeams).astype(np.float32))

    def scan_one(pose):
        return jax.vmap(lambda a: trace(pose[0], pose[1], pose[2] + a))(off)

    return jax.jit(jax.vmap(scan_one))


def main():
    dev = str(jax.devices()[0])
    kind = "gpu" if any(k in dev.lower() for k in ("cuda", "gpu", "metal")) else "cpu"

    sb_sim = make_scan(SIM_BEAMS)
    sb_pf = make_scan(PF_BEAMS)
    P_sim = jnp.asarray(np.stack([wx[:1], wy[:1], wth[:1]], 1))           # ONE scan
    P_pf = jnp.asarray(np.stack([wx, wy, wth], 1))                        # 4000
    sb_sim(P_sim).block_until_ready()
    sb_pf(P_pf).block_until_ready()

    sim_ms = timeit(lambda: sb_sim(P_sim).block_until_ready(), 50)
    pf_ms = timeit(lambda: sb_pf(P_pf).block_until_ready(), 20)

    print(f"\ndevice: {dev}  ({platform.system()} {platform.machine()})")
    print(f"{'backend':<22}{'SIM ms':>9}{'SIM hr':>8}{'PF ms':>9}{'PF hr':>8}")
    name = f"jax-rm({kind})"
    print(f"{name:<22}{sim_ms:>9.3f}{BUDGET_MS/sim_ms:>7.0f}x{pf_ms:>9.3f}{BUDGET_MS/pf_ms:>7.1f}x")
    print(f"\nPaste into results/<device>.md as the {name} row.")


if __name__ == "__main__":
    main()

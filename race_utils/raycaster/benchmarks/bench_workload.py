#!/usr/bin/env python3
"""
Real-time workload verification for the two consumers of the raycaster:

  SIM : 2 LiDARs x 40 Hz x 2200 beams        -> must finish 2 scans within 25 ms
  PF  : 4000 particles x 100 beams x 40 Hz   -> must finish the 4000x100 batch within 25 ms

For each method we measure the time of ONE real-time cycle (25 ms budget @ 40 Hz)
and report headroom = budget / measured. Headroom Nx => still real-time on
hardware up to ~Nx slower (the laptop / low-end question).

Run: /tmp/rlbench/bin/python bench_workload.py            (GPU jax)
     JAX_PLATFORMS=cpu /tmp/rlbench/bin/python bench_workload.py   (CPU jax)
"""
import os, sys, time, importlib.util
import numpy as np
from PIL import Image
import yaml
from scipy.ndimage import distance_transform_edt

CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
MAP_DIR = CAC + "/stack_master/maps/test"; MAP_YAML = MAP_DIR + "/test.yaml"
GYM_DIR = CAC + "/simulator/f1tenth_gym/gym"; RL = CAC + "/slam/range_libc/pywrapper"

FOV, MAXR = 4.7, 10.0
SIM_BEAMS, SIM_LIDARS = 2200, 2
PF_PARTICLES, PF_BEAMS = 4000, 100
HZ = 40; BUDGET = 1000.0 / HZ                      # 25 ms
THETA_DISC = 112

meta = yaml.safe_load(open(MAP_YAML)); res = float(meta["resolution"]); origin = meta["origin"]
img = np.array(Image.open(os.path.join(MAP_DIR, meta["image"])))
if img.ndim == 3: img = img[..., 0]
H, W = img.shape; MAXR_PX = MAXR / res
occupied = (img <= 128); free = ~occupied
clear = distance_transform_edt(free); ys, xs = np.where(clear > 5)
rngp = np.random.default_rng(0)
sel = rngp.choice(len(xs), size=PF_PARTICLES, replace=True)
col = xs[sel].astype(np.float32); row = ys[sel].astype(np.float32)
th = rngp.uniform(-np.pi, np.pi, PF_PARTICLES).astype(np.float32)
wx = (origin[0] + col * res).astype(np.float64)
wy = (origin[1] + (H - 1 - row) * res).astype(np.float64); wth = th.astype(np.float64)

def timeit(fn, reps):
    fn()                                            # warmup
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms per call

rows = []   # (method, sim_ms, pf_ms)

# ---------------- numba ----------------
spec = importlib.util.spec_from_file_location("laser_models", GYM_DIR + "/f110_gym/envs/laser_models.py")
lm = importlib.util.module_from_spec(spec); sys.modules["laser_models"] = lm; spec.loader.exec_module(lm)
sim_s = lm.ScanSimulator2D(SIM_BEAMS, FOV, max_range=MAXR); sim_s.set_map(MAP_YAML, ".pgm")
sim_p = lm.ScanSimulator2D(PF_BEAMS, FOV, max_range=MAXR); sim_p.set_map(MAP_YAML, ".pgm")
sim_s.scan(np.array([wx[0], wy[0], wth[0]]), None); sim_p.scan(np.array([wx[0], wy[0], wth[0]]), None)
def numba_sim():
    for k in range(SIM_LIDARS): sim_s.scan(np.array([wx[k], wy[k], wth[k]]), None)
def numba_pf():
    for i in range(PF_PARTICLES): sim_p.scan(np.array([wx[i], wy[i], wth[i]]), None)
rows.append(("numba f1tenth_gym (CPU)", timeit(numba_sim, 50), timeit(numba_pf, 3)))

# ---------------- range_libc ----------------
sys.path.insert(0, RL); import range_libc
oMap = range_libc.PyOMap(occupied)
ang_sim = np.linspace(-FOV/2, FOV/2, SIM_BEAMS).astype(np.float32)
ang_pf  = np.linspace(-FOV/2, FOV/2, PF_BEAMS).astype(np.float32)
parts = np.zeros((PF_PARTICLES, 3), np.float32); parts[:, 0] = col; parts[:, 1] = row; parts[:, 2] = th
out_pf = np.zeros(PF_PARTICLES * PF_BEAMS, np.float32)
qs = np.zeros((SIM_BEAMS, 3), np.float32); out_s = np.zeros(SIM_BEAMS, np.float32)
def rl_factory(ctor):
    m = ctor()
    def sim():
        for k in range(SIM_LIDARS):
            qs[:, 0] = col[k]; qs[:, 1] = row[k]; qs[:, 2] = th[k] + ang_sim
            m.calc_range_many(qs, out_s)
    def pf():
        m.calc_range_repeat_angles(parts, ang_pf, out_pf)   # 4000 poses x 100 angles, C++ batch
    return sim, pf
for nm, ctor in [("range_libc rm", lambda: range_libc.PyRayMarching(oMap, MAXR_PX)),
                 ("range_libc cddt", lambda: range_libc.PyCDDTCast(oMap, MAXR_PX, THETA_DISC)),
                 ("range_libc pcddt", lambda: (lambda m: (m.prune(), m)[1])(range_libc.PyCDDTCast(oMap, MAXR_PX, THETA_DISC))),
                 ("range_libc glt", lambda: range_libc.PyGiantLUTCast(oMap, MAXR_PX, THETA_DISC))]:
    s, p = rl_factory(ctor)
    rows.append((nm, timeit(s, 50), timeit(p, 20)))

# ---------------- jax ----------------
try:
    import jax, jax.numpy as jnp
    dev = str(jax.devices()[0])
    mflip = np.flipud(img); free2 = (mflip > 128).astype(np.float32)
    dt = jnp.asarray((distance_transform_edt(free2) * res).astype(np.float32))
    ox, oy = origin[0], origin[1]
    def lookup(x, y):
        c = jnp.clip(((x - ox)/res).astype(jnp.int32), 0, W-1); r = jnp.clip(((y - oy)/res).astype(jnp.int32), 0, H-1)
        return dt[r, c]
    def trace(px, py, ang):
        cs, sn = jnp.cos(ang), jnp.sin(ang); d0 = lookup(px, py)
        def cond(st): d, tot, x, y, it = st; return (d > 1e-4) & (tot <= MAXR) & (it < 128)
        def body(st): d, tot, x, y, it = st; x2 = x+d*cs; y2 = y+d*sn; d2 = lookup(x2, y2); return (d2, tot+d2, x2, y2, it+1)
        d, tot, *_ = jax.lax.while_loop(cond, body, (d0, d0, px, py, 0)); return jnp.minimum(tot, MAXR)
    def make_scan(nbeams):
        off = jnp.asarray(np.linspace(-FOV/2, FOV/2, nbeams).astype(np.float32))
        def scan_one(pose): return jax.vmap(lambda a: trace(pose[0], pose[1], pose[2]+a))(off)
        return jax.jit(jax.vmap(scan_one))
    sb_sim = make_scan(SIM_BEAMS); sb_pf = make_scan(PF_BEAMS)
    P_sim = jnp.asarray(np.stack([wx[:SIM_LIDARS], wy[:SIM_LIDARS], wth[:SIM_LIDARS]], 1).astype(np.float32))
    P_pf  = jnp.asarray(np.stack([wx, wy, wth], 1).astype(np.float32))
    sb_sim(P_sim).block_until_ready(); sb_pf(P_pf).block_until_ready()
    def jsim(): sb_sim(P_sim).block_until_ready()
    def jpf():  sb_pf(P_pf).block_until_ready()
    rows.append((f"jax DT-march ({dev.split(':')[0] if ':' in dev else dev})", timeit(jsim, 50), timeit(jpf, 20)))
except Exception as e:
    print("jax skipped:", e)

# ---------------- report ----------------
print(f"\nReal-time budget @ {HZ} Hz = {BUDGET:.1f} ms/cycle")
print(f"SIM  workload: {SIM_LIDARS} lidars x {SIM_BEAMS} beams  ({SIM_LIDARS*SIM_BEAMS*HZ:,} rays/s)")
print(f"PF   workload: {PF_PARTICLES} particles x {PF_BEAMS} beams  ({PF_PARTICLES*PF_BEAMS*HZ:,} rays/s)\n")
print(f"{'method':<28}{'SIM ms':>8}{'SIM x':>7}{'SIM':>5}   {'PF ms':>8}{'PF x':>7}{'PF':>5}")
print("-" * 76)
for nm, sm, pm in rows:
    sv = "OK" if sm < BUDGET else "FAIL"; pv = "OK" if pm < BUDGET else "FAIL"
    print(f"{nm:<28}{sm:>8.3f}{BUDGET/sm:>6.0f}x{sv:>5}   {pm:>8.3f}{BUDGET/pm:>6.1f}x{pv:>5}")
print("\n(x = headroom = budget/time; higher = survives slower hardware)")

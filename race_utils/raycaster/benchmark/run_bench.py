#!/usr/bin/env python3
"""
Raycaster benchmark — consistent, environment-tagged.

Measures the TWO real workloads of the unicorn raycaster on whatever backends are
available on THIS machine, and records WHICH environment produced the numbers so
results from the 4080 desktop, an Orin, a laptop, etc. stay comparable:

  SIM : time to generate ONE LiDAR scan          (1 lidar x SIM_BEAMS)   -> latency
  PF  : time for the particle-filter batch        (PF_PARTICLES x PF_BEAMS) -> throughput

Backends are auto-detected; anything not built/available on this host is printed
as N/A with how to enable it (see README.md for the full target matrix).

Usage:
    python run_bench.py                 # uses map 'test', writes results/<device>.md
    MAP=f THETA_DISC=112 python run_bench.py
    python run_bench.py --no-write      # print only

range_libc must be importable (the pybind11 build:
`pip install --no-build-isolation -e ../range_libc/pywrapper`).
jax is optional and usually lives in a SEPARATE venv (see bench_jax.py / README).
"""
import os
import sys
import time
import platform
import subprocess
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))   # unicorn-racing-stack

# ----- workload definition (keep in sync with README) ------------------------
FOV, MAXR = 4.7, 10.0
SIM_BEAMS = 1080            # one lidar frame
PF_PARTICLES, PF_BEAMS = 4000, 100
HZ = 40
BUDGET_MS = 1000.0 / HZ     # 25 ms real-time budget @ 40 Hz
THETA_DISC = int(os.environ.get("THETA_DISC", 112))
MAPNAME = os.environ.get("MAP", "test")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def cpu_name():
    if platform.system() == "Linux":
        for line in sh("grep -m1 'model name' /proc/cpuinfo").splitlines():
            return line.split(":", 1)[-1].strip()
        return sh("grep -m1 'Model' /proc/cpuinfo").split(":")[-1].strip() or platform.machine()
    if platform.system() == "Darwin":
        return sh("sysctl -n machdep.cpu.brand_string") or platform.machine()
    return platform.processor() or platform.machine()


def gpu_name():
    n = sh("nvidia-smi --query-gpu=name --format=csv,noheader")
    if n:
        return f"NVIDIA {n.splitlines()[0]}" if not n.startswith("NVIDIA") else n.splitlines()[0]
    if platform.system() == "Darwin":
        return sh("system_profiler SPDisplaysDataType | grep -m1 Chipset | cut -d: -f2").strip() or "Apple GPU"
    return "none"


def environment():
    import range_libc
    try:
        rl_cuda = bool(range_libc.SHOULD_USE_CUDA)
    except Exception:
        rl_cuda = False
    env = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": cpu_name(),
        "cores": f"{os.cpu_count()} threads",
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "range_libc": "pybind11/cuda" if rl_cuda else "pybind11/cpu",
    }
    return env


def timeit(fn, reps):
    fn()                                    # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3      # ms/call


# ----- map -> occupancy (pixel space; world_scale defaults to 1, fine for timing)
def load_map():
    import yaml
    from PIL import Image
    map_dir = os.path.join(REPO, "stack_master", "maps", MAPNAME)
    meta = yaml.safe_load(open(os.path.join(map_dir, f"{MAPNAME}.yaml")))
    img = np.array(Image.open(os.path.join(map_dir, meta["image"])))
    if img.ndim == 3:
        img = img[..., 0]
    occupied = (img <= 128)
    return occupied, float(meta.get("resolution", 0.05)), img.shape


def sample_poses(occupied, n, seed=0):
    from scipy.ndimage import distance_transform_edt
    clear = distance_transform_edt(~occupied)
    ys, xs = np.where(clear > 5)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(xs), size=n, replace=True)
    col = xs[sel].astype(np.float32)
    row = ys[sel].astype(np.float32)
    th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    return col, row, th


# ----- range_libc backends ---------------------------------------------------
def rangelib_rows(occupied, res):
    import range_libc
    maxr_px = MAXR / res
    omap = range_libc.PyOMap(occupied)
    col, row, th = sample_poses(occupied, PF_PARTICLES)

    ang_sim = np.linspace(-FOV / 2, FOV / 2, SIM_BEAMS).astype(np.float32)
    ang_pf = np.linspace(-FOV / 2, FOV / 2, PF_BEAMS).astype(np.float32)
    parts = np.zeros((PF_PARTICLES, 3), np.float32)
    parts[:, 0] = col; parts[:, 1] = row; parts[:, 2] = th
    out_pf = np.zeros(PF_PARTICLES * PF_BEAMS, np.float32)
    qs = np.zeros((SIM_BEAMS, 3), np.float32)
    out_s = np.zeros(SIM_BEAMS, np.float32)

    def mk(name, ctor):
        m = ctor()
        def sim():                      # ONE scan: 1 lidar x SIM_BEAMS
            qs[:, 0] = col[0]; qs[:, 1] = row[0]; qs[:, 2] = th[0] + ang_sim
            m.calc_range_many(qs, out_s)
        def pf():                       # 4000 poses x PF_BEAMS, single C++ batch call
            m.calc_range_repeat_angles(parts, ang_pf, out_pf)
        return name, sim, pf

    def pcddt():
        m = range_libc.PyCDDTCast(omap, maxr_px, THETA_DISC); m.prune(); return m

    specs = [
        ("rangelib(bl, cpu)",    lambda: range_libc.PyBresenhamsLine(omap, maxr_px)),
        ("rangelib(rm, cpu)",    lambda: range_libc.PyRayMarching(omap, maxr_px)),
        ("rangelib(cddt, cpu)",  lambda: range_libc.PyCDDTCast(omap, maxr_px, THETA_DISC)),
        ("rangelib(pcddt, cpu)", pcddt),
        ("rangelib(GLUT, cpu)",  lambda: range_libc.PyGiantLUTCast(omap, maxr_px, THETA_DISC)),
    ]
    rows = []
    for name, ctor in specs:
        n, sim, pf = mk(name, ctor)
        rows.append((n, timeit(sim, 100), timeit(pf, 20)))
    return rows


# ----- backends not available from a plain CPU pybind build ------------------
# (kept here so the printed table always shows the full target matrix)
NOT_AVAILABLE = [
    ("rangelib(bl, cpu-omp)",   "build range_libc with -fopenmp"),
    ("rangelib(rm, cpu-omp)",   "build range_libc with -fopenmp"),
    ("rangelib(cddt, cpu-omp)", "build range_libc with -fopenmp"),
    ("rangelib(pcddt, cpu-omp)","build range_libc with -fopenmp"),
    ("rangelib(GLUT, cpu-omp)", "build range_libc with -fopenmp"),
    ("rangelib(rmgpu, cuda)",   "build range_libc WITH_CUDA=ON (needs nvcc)"),
    ("rangelib(rmgpu, apple)",  "Apple GPU device (Metal) required"),
    ("rangelib(rmgpu, intel)",  "Intel GPU device required"),
    ("jax-rm(cpu)",             "pip install jax  (separate venv) + run bench_jax.py"),
    ("jax-rm(gpu, cuda)",       "pip install 'jax[cuda12]' on an NVIDIA host"),
    ("jax-rm(gpu, apple)",      "jax-metal on Apple Silicon"),
    ("jax-rm(gpu, intel)",      "jax + Intel GPU (oneAPI) host"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    env = environment()
    occupied, res, shape = load_map()

    rows = rangelib_rows(occupied, res)

    # optional: numba & jax if importable in this interpreter
    extra = []
    # (jax usually runs from bench_jax.py in its own venv; left to NOT_AVAILABLE here)

    lines = []
    p = lines.append
    p(f"# Raycaster benchmark — {env['host']}")
    p("")
    p("## Environment")
    p("")
    p("| field | value |")
    p("|---|---|")
    for k in ("host", "os", "arch", "cpu", "cores", "gpu", "python", "numpy", "range_libc"):
        p(f"| {k} | {env[k]} |")
    p(f"| map | {MAPNAME} {shape[1]}x{shape[0]} px @ {res} m/px |")
    p(f"| workload | SIM = 1 scan x {SIM_BEAMS} beams; PF = {PF_PARTICLES} particles x {PF_BEAMS} beams |")
    p(f"| budget | {BUDGET_MS:.1f} ms/cycle @ {HZ} Hz |")
    p("")
    p("## Results")
    p("")
    p("`headroom = budget / time` — survives ~Nx slower hardware. SIM is one scan; PF is the full 4000-particle batch.")
    p("")
    p("| backend | SIM ms | SIM headroom | PF ms | PF headroom |")
    p("|---|--:|--:|--:|--:|")
    for name, sm, pm in rows:
        p(f"| {name} | {sm:.3f} | {BUDGET_MS/sm:.0f}x | {pm:.3f} | {BUDGET_MS/pm:.1f}x |")
    for name, why in NOT_AVAILABLE:
        p(f"| {name} | N/A | — | N/A | — |  <!-- {why} -->")
    p("")
    p("N/A rows are not buildable/available on this host — see comments and README.md.")

    out = "\n".join(lines)
    print(out)

    if not args.no_write:
        tag = (env["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
               .replace(" ", "_") or env["cpu"].split()[0])
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        path = os.path.join(HERE, "results", f"{tag}.md")
        with open(path, "w") as f:
            f.write(out + "\n")
        print(f"\n[written] {os.path.relpath(path, REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()

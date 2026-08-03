# Raycaster benchmark

The 2D-LiDAR simulator and the particle filter are both **raycaster-bound**, so
which raycaster we ship decides real-time feasibility. This directory measures
every backend on the **two real workloads**, **tags each result with the machine
it ran on**, and keeps the numbers comparable across the 4080 desktop, an Orin,
a laptop, etc.

> Migrated and consolidated from the `raycast_test/` scratch tree. range_libc is
> now the in-repo **pybind11** build (`../range_libc/pywrapper`), not the old
> Cython/CAC copy.

## The two workloads

| consumer | pattern | what we measure | bound by |
|---|---|---|---|
| **SIM** | 1 LiDAR scan | time to generate **one** `SIM_BEAMS`-beam scan | latency (≤25 ms @ 40 Hz) |
| **PF** | particle filter | time for **4000 particles × 100 beams** in one batch | throughput |

`headroom = budget / time` (budget = 25 ms @ 40 Hz). Headroom *N×* ⇒ still
real-time on hardware up to ~*N×* slower — that's the "will it run on the Orin /
a weak laptop" number. **SIM** is latency-bound and trivially safe everywhere;
**PF** (≈90× heavier) is the real constraint.

> 1 scan = one full `SIM_BEAMS`-beam frame (default 1080). rays/s = scans/s × beams.

## How to run (consistent + environment-tagged)

```bash
conda activate unicorn          # provides the pybind11 range_libc
# build range_libc once if you haven't (header-only pybind, no numpy at build):
pip install --no-build-isolation -e ../range_libc/pywrapper

cd race_utils/raycaster/benchmark
python run_bench.py             # map 'test'; writes results/<gpu-or-cpu>.md
MAP=f THETA_DISC=112 python run_bench.py
```

`run_bench.py` auto-detects the host (CPU / GPU / OS / arch / range_libc build),
runs every **available** backend, and writes a per-device file under `results/`
so re-running on a new machine never overwrites another's numbers. Backends that
can't run on the host are printed as **N/A** with how to enable them.

**jax** is measured separately (it usually needs its own venv to avoid disturbing
the ROS numpy):

```bash
python -m venv /tmp/jaxbench && /tmp/jaxbench/bin/pip install -U "jax[cuda12]" numpy pillow pyyaml scipy
/tmp/jaxbench/bin/python bench_jax.py            # GPU
JAX_PLATFORMS=cpu /tmp/jaxbench/bin/python bench_jax.py   # CPU
```

## Target backend matrix

What we want filled in, per device. `run_bench.py` fills the rangelib CPU rows
automatically; the rest need the noted build/device and are marked N/A until then.

| backend | how it's enabled | typical host |
|---|---|---|
| `rangelib(bl/rm/cddt/pcddt/GLUT, cpu)` | default pybind build | any |
| `rangelib(…, cpu-omp)` | rebuild range_libc with `-fopenmp` | multi-core CPU |
| `rangelib(rmgpu, cuda)` | `WITH_CUDA=ON` range_libc build (nvcc) | NVIDIA (4080 / Orin) |
| `rangelib(rmgpu, apple)` | Metal build | Apple Silicon |
| `rangelib(rmgpu, intel)` | Intel GPU build | Intel Arc/iGPU |
| `jax-rm(cpu)` | `pip install jax` | any |
| `jax-rm(gpu, cuda)` | `jax[cuda12]` | NVIDIA |
| `jax-rm(gpu, apple)` | `jax-metal` | Apple Silicon |
| `jax-rm(gpu, intel)` | jax + oneAPI | Intel GPU |

## Results by environment

Each file in [`results/`](results/) carries its own environment table. Append a
new device by running on it; do **not** edit another device's file.

- [`results/RTX_4080_SUPER.md`](results/RTX_4080_SUPER.md) — Ryzen 9 7950X + RTX 4080 SUPER (CPU backends)
- _Orin: TODO — run `run_bench.py` on the Orin and commit `results/Orin_*.md`._

### Snapshot — RTX 4080 SUPER (test map, 1080-beam SIM, 4000×100 PF)

| backend | SIM ms | SIM headroom | PF ms | PF headroom | note |
|---|--:|--:|--:|--:|---|
| `rangelib(GLUT, cpu)`  | 0.008 | 3247× | **1.88** | **13×** | fastest; big init + LUT memory |
| `rangelib(pcddt, cpu)` | 0.015 | 1641× | 8.03 | 3.1× | **balanced — PF default** |
| `rangelib(cddt, cpu)`  | 0.015 | 1682× | 8.47 | 3.0× | fast init |
| `rangelib(rm, cpu)`    | 0.021 | 1214× | 21.2 | 1.2× | simple/robust; **sim default** |
| `rangelib(bl, cpu)`    | 0.338 | 74×   | 125  | 0.2× | slow, not recommended |

(jax / cuda / omp rows: N/A on this run — see the target matrix.)

## Reading it

- **SIM**: every backend has 100×+ headroom → pick the raycaster for *code unity*
  (one engine for sim + PF), not speed. The repo sim default is `rm`.
- **PF**: throughput-bound, the real limit. On the 4080 desktop:
  - `GLUT` (13×) → safe even on much weaker hardware (cost: ~0.7 s init + LUT memory).
  - `pcddt`/`cddt` (~3×) → fine on desktop, marginal on a weak laptop.
  - `rm` (1.2×) → desktop-only for the full 4000-particle batch.
  - `bl` → fails PF, don't use.
  - For low-end / Orin guarantees, prefer **`GLUT`** for PF, or reduce particles/beams.

## Algorithms (one line each)

| backend | principle |
|---|---|
| `bl` Bresenham/DDA | integer cell-stepping along the grid |
| `rm` ray-marching (= sphere tracing) | jump by the precomputed distance-transform each step (adaptive) |
| `cddt`/`pcddt` | discretize directions, compressed LUT of obstacle distances; `p` = pruned |
| `GLUT` GiantLUT | full (x,y,θ)→range table, O(1) lookup; biggest memory/init |
| `jax-rm` | same DT sphere-tracing, on XLA — wins only **GPU + large batch** |

`rm`, `jax-rm` and numba are the *same* DT sphere-tracing algorithm; `cddt`/`GLUT`
precompute past it. All backends agree on measured ranges (validated in
`../range_libc/pywrapper` cross-checks: rm ≈ cddt ≈ GLUT).

## References

- range_libc / CDDT: Walsh & Karaman, *"CDDT: Fast Approximate 2D Ray Casting…"*, ICRA 2018, [arXiv:1705.01167](https://arxiv.org/abs/1705.01167)
- F1TENTH gym: O'Kelly et al., NeurIPS 2019 — [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)
- JAX raycaster: [f1tenth/f1tenth_gym_jax](https://github.com/f1tenth/f1tenth_gym_jax)
- Sphere tracing: Hart, *The Visual Computer*, 1996

# SETUP — reproduce the raycaster benchmarks

For re-running on a new machine (NUC, Mac mini, laptop) and recording a new
`results/<date>_<machine>.md`.

## Dependencies (from `creating_autonomous_car`)
The benchmarks reuse CAC's assets — set `CAC_DIR` to its path:
```bash
export CAC_DIR=/path/to/creating_autonomous_car
```
- `f110_gym` (numba sim)         : `$CAC_DIR/simulator/f1tenth_gym/gym`
- `range_libc` (C++/Cython)      : `$CAC_DIR/slam/range_libc/pywrapper`
- maps                           : `$CAC_DIR/stack_master/maps/{test,f}`

## Python env
range_libc's `setup.py` uses `Cython.Distutils.old_build_ext` (Cython <3) and
`distutils` (gone in py3.12 → needs `setuptools<81`).
```bash
python3.12 -m venv ~/rlbench && . ~/rlbench/bin/activate
pip install -U pip
pip install "numpy<2" "cython<3" "setuptools<81" numba scipy pillow pyyaml transforms3d
```

## Build range_libc (CPU)
```bash
cd "$CAC_DIR/slam/range_libc/pywrapper"
WITH_CUDA=OFF python setup.py build_ext --inplace      # -> range_libc.*.so
```
Import it by adding the pywrapper dir to `PYTHONPATH` (the scripts do this).
- NVIDIA + matching CUDA toolkit: `WITH_CUDA=ON CUDAHOME=/usr/local/cuda ...` adds `rmgpu`.
  (Skipped on this desktop: CUDA 13 vs the old kernels.)
- Apple Silicon / Intel: CPU build only (no CUDA). C++ compiles fine on ARM/x86.

## jax (optional — only useful on NVIDIA GPU)
```bash
pip install "jax[cuda13]>=0.7.2,<0.8"     # NVIDIA
# CPU-only (Mac/Intel/no-GPU):  pip install "jax>=0.7.2,<0.8"
# Apple Silicon Metal (experimental, unreliable): pip install jax-metal
```
> The jax raycaster here is a self-contained reimplementation of `jax_pf`'s
> distance-transform sphere-tracing — no external `jax_pf` install needed.

## Run
```bash
MAP=test            $PY benchmarks/bench_raycast.py     # grid + segment (MAP=test|f)
                    $PY benchmarks/bench_jax.py          # jax GPU
JAX_PLATFORMS=cpu   $PY benchmarks/bench_jax.py          # jax CPU
                    $PY benchmarks/bench_workload.py     # SIM vs PF real-time check
```
If numba complains about a stale cache (`<dynamic>` module): `NUMBA_CACHE_DIR=/tmp/nc $PY ...`.

## Record a result
Copy `results/2026-06-17_ryzen7950x_rtx4080super.md` to a new
`results/<YYYY-MM-DD>_<machine>.md`, paste the three tables, and update the
Environment block (CPU/GPU/OS/lib versions). That builds the cross-platform history.

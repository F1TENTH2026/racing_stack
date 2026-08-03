# range_libc fixed + modernized for Jazzy — 2026-06-18

Machine: Ryzen 9 7950X + RTX 4080 SUPER (see `2026-06-17_*` for full env).
Build: **Cython 3.2.5, numpy 2.4.6, Python 3.12, setuptools** (no `old_build_ext`).

## The bug (and fix)
range_libc's numpy `PyOMap` constructor (`RangeLibc.pyx`) wrote `grid[x][y]=arr[y,x]`
(**transposed** vs the working ROS `OccupancyGrid` path `grid[x][y]=arr[x,y]`) and did
**not set the world transform**. Result: degenerate/wrong scans on real maps (transposed;
out-of-bounds on non-square). Fixed: fill grid like the ROS path **and** set
`world_scale=resolution`, `world_origin`, `world_angle` → `calc_range` works in **world
meters** with standard convention (θ=0→+x, θ=90°→+y), non-square safe.

Verified on a non-square 80×120 map: +x wall 2.50 m ✓, +y wall 1.50 m ✓, open 30 m ✓.

## Modernization for ROS 2 Jazzy
- `setup.py`: `cythonize()` + `setuptools` (was `Cython.Distutils.old_build_ext`, removed in Cython 3; `distutils` gone in py3.12). c++17.
- `RangeLibc.pyx`: `xrange`→`range`; numpy-2 `PyOMap` ctor accepts `resolution/origin_x/origin_y`.
- Builds clean with **Cython 3 + numpy 2 + py3.12**, both CPU and CUDA.

## All backends vs numba ScanSimulator2D — `f` map, 1080 beams, 300 poses

| backend | sim MAE | within 10cm | ms/scan | vs numba | notes |
|---|--:|--:|--:|--:|---|
| numba (reference) | — | — | 0.0738 | 1.0× | existing sim |
| **rm** (RayMarching) | **2.9 cm** | 99.0% | 0.0576 | 1.3× | exact DT, no precompute |
| cddt | 9.0 cm | 83% | 0.0376 | 2.0× | low init/memory |
| pcddt | 9.0 cm | 83% | 0.0344 | 2.1× | pruned (PF default) |
| **glt** | 8.5 cm | 85% | 0.0122 | **6.1×** | fastest CPU, high memory |
| bl | 4.8 cm | 95% | 0.1540 | 0.5× | slow |
| lut (TD=720) | 1.8 cm | 97% | 0.0267 | 2.8× | portable, numba-oracle |
| **rmgpu** (GPU) | **2.9 cm** | 99.1% | — | — | **PF 4000×60 = 0.31 ms (≈80× real-time)** |

`cddt/pcddt/glt` ~9 cm = their `theta_disc=112` angle quantization (raise it for less error,
more memory). `rm`/`rmgpu` use exact DT marching → match numba to ~3 cm.

## Takeaways
- range_libc now usable directly: **pcddt** (low memory), **glt** (fastest CPU),
  **rmgpu** (GPU — best for PF batches at ~80× real-time), **rm** (most accurate CPU).
- Build CPU by default; **`WITH_CUDA=ON` for rmgpu** (compiled fine with CUDA 13).

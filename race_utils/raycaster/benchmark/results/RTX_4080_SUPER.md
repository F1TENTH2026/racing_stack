# Raycaster benchmark — 4080s

## Environment

| field | value |
|---|---|
| host | 4080s |
| os | Linux 6.17.0-29-generic |
| arch | x86_64 |
| cpu | AMD Ryzen 9 7950X 16-Core Processor |
| cores | 32 threads |
| gpu | NVIDIA GeForce RTX 4080 SUPER |
| python | 3.12.13 |
| numpy | 2.4.6 |
| range_libc | pybind11/cpu |
| map | test 351x358 px @ 0.05 m/px |
| workload | SIM = 1 scan x 1080 beams; PF = 4000 particles x 100 beams |
| budget | 25.0 ms/cycle @ 40 Hz |

## Results

`headroom = budget / time` — survives ~Nx slower hardware. SIM is one scan; PF is the full 4000-particle batch.

| backend | SIM ms | SIM headroom | PF ms | PF headroom |
|---|--:|--:|--:|--:|
| rangelib(bl, cpu) | 0.338 | 74x | 125.432 | 0.2x |
| rangelib(rm, cpu) | 0.021 | 1214x | 21.184 | 1.2x |
| rangelib(cddt, cpu) | 0.015 | 1682x | 8.465 | 3.0x |
| rangelib(pcddt, cpu) | 0.015 | 1641x | 8.027 | 3.1x |
| rangelib(GLUT, cpu) | 0.008 | 3247x | 1.877 | 13.3x |
| rangelib(bl, cpu-omp) | N/A | — | N/A | — |  <!-- build range_libc with -fopenmp -->
| rangelib(rm, cpu-omp) | N/A | — | N/A | — |  <!-- build range_libc with -fopenmp -->
| rangelib(cddt, cpu-omp) | N/A | — | N/A | — |  <!-- build range_libc with -fopenmp -->
| rangelib(pcddt, cpu-omp) | N/A | — | N/A | — |  <!-- build range_libc with -fopenmp -->
| rangelib(GLUT, cpu-omp) | N/A | — | N/A | — |  <!-- build range_libc with -fopenmp -->
| rangelib(rmgpu, cuda) | N/A | — | N/A | — |  <!-- build range_libc WITH_CUDA=ON (needs nvcc) -->
| rangelib(rmgpu, apple) | N/A | — | N/A | — |  <!-- Apple GPU device (Metal) required -->
| rangelib(rmgpu, intel) | N/A | — | N/A | — |  <!-- Intel GPU device required -->
| jax-rm(cpu) | N/A | — | N/A | — |  <!-- pip install jax  (separate venv) + run bench_jax.py -->
| jax-rm(gpu, cuda) | N/A | — | N/A | — |  <!-- pip install 'jax[cuda12]' on an NVIDIA host -->
| jax-rm(gpu, apple) | N/A | — | N/A | — |  <!-- jax-metal on Apple Silicon -->
| jax-rm(gpu, intel) | N/A | — | N/A | — |  <!-- jax + Intel GPU (oneAPI) host -->

N/A rows are not buildable/available on this host — see comments and README.md.

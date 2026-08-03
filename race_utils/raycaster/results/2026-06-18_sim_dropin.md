# Simulator drop-in comparison — 2026-06-18

RaycastEngine vs the **existing** simulator raycaster (f1tenth_gym numba
`ScanSimulator2D`). Same map / poses / 1080 beams. Machine: Ryzen 9 7950X
(see `2026-06-17_ryzen7950x_rtx4080super.md` for full environment).

## `f` map, 1080 beams, fov 4.7, max 10 m, 300 poses

| backend | vs existing numba | per-scan | speed |
|---|---|--:|--:|
| existing numba `ScanSimulator2D` | — (reference) | 0.0744 ms | 13,437 /s |
| **RaycastEngine `lut` (TD=720, numba oracle)** | **MAE 1.7 cm, p95 6.2 cm, 97.7% within 10 cm** | **0.0269 ms** | **37,226 /s (2.8×)** |

→ The unified engine produces the **same scan as the current simulator** (1.7 cm =
720-bin angle quantization; raise `theta_disc` for less) at **2.8× the speed**, and
the `lut` backend needs **no range_libc at query time** (portable).

## Notes
- The `lut` table is built from the numba `ScanSimulator2D` oracle → it matches the
  existing simulator's coordinate frame *by construction* (no convention guessing).
- range_libc direct backends (pcddt/cddt/glt) are ~3–10× faster but their **numpy
  `PyOMap` mis-handles real (flipped / non-square) maps** (degenerate scans). Fixing
  that needs range_libc's PNG/ROS-grid map path; for now the verified drop-in is `lut`.
- LUT memory ∝ W·H·theta_disc·2 B. `f` 600×600: TD=112 ≈ 80 MB, TD=720 ≈ 518 MB raw
  (compresses heavily on disk). Pick TD per use: 112 PF-grade, 720 sim-grade.

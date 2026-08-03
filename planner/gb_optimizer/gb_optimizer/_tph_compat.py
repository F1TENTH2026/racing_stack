"""Compatibility shims for trajectory_planning_helpers (tph).

tph==0.79 (latest on PyPI) was written against scipy ~1.3-1.6 / numpy ~1.18.
On the RoboStack env (scipy 1.17, numpy 2.x) two things changed under it inside
spline_approximation():

  1. scipy.interpolate.splev now returns a 2-D array, so
     dist_to_p -> spatial.distance.euclidean(p, s)
     raises "Input vector should be 1-D".

  2. scipy.optimize.fmin now returns a shape-(1,) array, so
     closest_t_glob_cl[i] = optimize.fmin(...)
     raises "setting an array element with a sequence".

Both are fixed here without touching the installed third-party files: we
replace dist_to_p, and wrap the `optimize` module reference that the function
closes over so fmin yields a scalar for scalar x0. Idempotent, import-time only.
"""

import numpy as np
from scipy import interpolate, optimize


class _OptimizeShim:
    """Proxy for scipy.optimize that squeezes fmin's length-1 result back to a
    scalar (old scipy behaviour) so it fits a scalar array slot."""

    def fmin(self, *args, **kwargs):
        res = optimize.fmin(*args, **kwargs)
        res = np.asarray(res)
        return res.item() if res.size == 1 else res

    def __getattr__(self, name):
        return getattr(optimize, name)


def _patch_spline_approximation():
    try:
        from trajectory_planning_helpers import spline_approximation as sa
    except Exception:
        return  # tph not importable yet -> nothing to patch

    if getattr(sa, "_unicorn_patched", False):
        return

    def dist_to_p(t_glob, path, p):
        # splev may return a 2-D array under modern scipy; ravel both operands.
        s = np.ravel(interpolate.splev(t_glob, path))
        return float(np.linalg.norm(np.ravel(p) - s))

    sa.dist_to_p = dist_to_p
    sa.optimize = _OptimizeShim()   # the function body calls optimize.fmin(...)
    sa._unicorn_patched = True


_patch_spline_approximation()

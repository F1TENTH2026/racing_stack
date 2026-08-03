"""gb_optimizer package init.

Applies a runtime compatibility shim for trajectory_planning_helpers so the
global optimizer works against the modern scipy/numpy shipped by RoboStack.
Importing any gb_optimizer module triggers this, so no edit to the installed
third-party package (and no build-time patch step) is required.
"""

from . import _tph_compat  # noqa: F401  (side-effect: patches tph on import)

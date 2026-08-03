"""
range_libc — pybind11 build (Python 3.12, header-only binding).

  pip install [-e] .            # the normal path; no Cython, no numpy at build
  python setup.py build_ext --inplace

The whole raycaster lives in includes/RangeLib.h; the only compiled dependency
is vendor/lodepng (OMap PNG load/save). GPU (rmgpu) needs a separate CUDA build
and is exposed as a stub here.
"""
import os
import platform

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = ["-O3", "-ffast-math", "-fno-math-errno", "-w"]

if platform.system().lower() == "darwin":
    # keep the wheel loadable on the building Mac's OS version
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", platform.mac_ver()[0] or "11.0")

ext_modules = [
    Pybind11Extension(
        "range_libc",
        ["range_libc_pybind.cpp", "../vendor/lodepng/lodepng.cpp"],
        include_dirs=["../"],          # so "includes/..." and "vendor/..." resolve
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="range_libc",
    version="0.3",
    author="Corey Walsh",
    description="Fast 2D raycasting (pybind11 binding)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

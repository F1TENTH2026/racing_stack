// range_libc_pybind.cpp
// -----------------------------------------------------------------------------
// Header-only pybind11 binding for range_libc. Drop-in replacement for the old
// Cython pywrapper (`RangeLibc.pyx`): exposes the same `range_libc` module with
// PyOMap / PyBresenhamsLine / PyRayMarching / PyCDDTCast / PyGiantLUTCast and
// the same method names.
//
// Why this exists: the Cython build required Cython + numpy AT BUILD TIME, and
// pip build-isolation pulled a fresh PyPI numpy that is broken on some macOS
// (Accelerate ILP64). pybind11/numpy.h carries its own array API, so this needs
// ONLY a C++17 compiler + pybind11 headers — no Cython, no numpy-at-build.
//
// All raycasting algorithms live in includes/RangeLib.h (header-only). The only
// compiled dependency is vendor/lodepng (OMap PNG load/save); gflags is dropped.
// -----------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

#include "includes/RangeLib.h"

namespace py = pybind11;
using namespace ranges;

// Inputs: lenient (forcecast makes a read-only temp if dtype/layout differ).
using f_in = py::array_t<float, py::array::c_style | py::array::forcecast>;
using d_in = py::array_t<double, py::array::c_style | py::array::forcecast>;
// Outputs: strict (no forcecast) so we fill the CALLER's buffer in place.
using f_out = py::array_t<float, py::array::c_style>;
using d_out = py::array_t<double, py::array::c_style>;

// CDDTCast.prune() defaults to the construction max_range; the old PyCDDTCast
// kept it in a python-side field. Subclass to carry it.
struct CDDTCastPy : CDDTCast {
    float py_max_range;
    CDDTCastPy(OMap m, float mr, unsigned int td)
        : CDDTCast(m, mr, td), py_max_range(mr) {}
};

// Construct an OMap, mirroring the old PyOMap.__cinit__ exactly.
static OMap* make_omap(py::object arg1, py::object arg2, float resolution,
                       float origin_x, float origin_y, float origin_theta) {
    OMap* m = nullptr;
    bool set_trans = false;

    if (py::isinstance<py::array>(arg1)) {
        // Occupancy array [H, W]; truthy == occupied. grid[row][col] = arr[row,col],
        // built with OMap(H, W) exactly like the old wrapper.
        auto occ = arg1.cast<d_in>();
        if (occ.ndim() != 2)
            throw std::runtime_error("PyOMap: occupancy array must be 2D [H, W]");
        const int H = static_cast<int>(occ.shape(0));
        const int W = static_cast<int>(occ.shape(1));
        auto u = occ.unchecked<2>();
        m = new OMap(H, W);
        for (int x = 0; x < H; ++x)
            for (int y = 0; y < W; ++y)
                if (u(x, y) != 0.0) m->grid[x][y] = true;
        m->world_scale = resolution;
        m->world_angle = origin_theta;
        m->world_origin_x = origin_x;
        m->world_origin_y = origin_y;
        m->world_sin_angle = std::sin(origin_theta);
        m->world_cos_angle = std::cos(origin_theta);
        set_trans = true;
    } else if (!arg1.is_none() && !arg2.is_none() &&
               py::isinstance<py::int_>(arg1) && py::isinstance<py::int_>(arg2)) {
        m = new OMap(arg1.cast<int>(), arg2.cast<int>());
    } else if (!arg1.is_none() && !arg2.is_none() && py::isinstance<py::str>(arg1)) {
        m = new OMap(arg1.cast<std::string>(), arg2.cast<float>());
    } else if (!arg1.is_none() && py::hasattr(arg1, "info") && py::hasattr(arg1, "data")) {
        // nav_msgs/OccupancyGrid, duck-typed so we keep no hard ROS dep in C++.
        py::object info = arg1.attr("info");
        const int W = info.attr("width").cast<int>();
        const int H = info.attr("height").cast<int>();
        m = new OMap(H, W);
        auto data = arg1.attr("data").cast<std::vector<int>>();
        for (int x = 0; x < H; ++x)
            for (int y = 0; y < W; ++y)
                if (data[static_cast<size_t>(x) * W + y] > 10) m->grid[x][y] = true;
        py::object o = info.attr("origin").attr("orientation");
        const double qx = o.attr("x").cast<double>(), qy = o.attr("y").cast<double>();
        const double qz = o.attr("z").cast<double>(), qw = o.attr("w").cast<double>();
        const double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                      1.0 - 2.0 * (qy * qy + qz * qz));
        const double angle = -yaw;  // matches old quaternion_to_angle() * -1
        py::object p = info.attr("origin").attr("position");
        m->world_scale = info.attr("resolution").cast<float>();
        m->world_angle = static_cast<float>(angle);
        m->world_origin_x = p.attr("x").cast<float>();
        m->world_origin_y = p.attr("y").cast<float>();
        m->world_sin_angle = static_cast<float>(std::sin(angle));
        m->world_cos_angle = static_cast<float>(std::cos(angle));
        set_trans = true;
    } else if (!arg1.is_none() && py::isinstance<py::str>(arg1)) {
        m = new OMap(arg1.cast<std::string>());
    } else {
        py::print("Failed to construct PyOMap, check argument types.");
        m = new OMap(1, 1);
    }

    if (!set_trans) {
        m->world_scale = 1.0f;     m->world_angle = 0.0f;
        m->world_origin_x = 0.0f;  m->world_origin_y = 0.0f;
        m->world_sin_angle = 0.0f; m->world_cos_angle = 1.0f;
    }
    return m;
}

// Bind the methods shared by every CPU range method on class `c`.
template <typename T, typename C>
static void bind_methods(C& c, bool has_calc_range, bool has_save_trace) {
    if (has_calc_range)
        c.def("calc_range", &T::calc_range, py::arg("x"), py::arg("y"), py::arg("heading"));

    c.def("calc_range_many",
          [](T& s, f_in ins, f_out outs) {
              s.numpy_calc_range(const_cast<float*>(ins.data()),
                                 outs.mutable_data(), static_cast<int>(outs.shape(0)));
          },
          py::arg("ins"), py::arg("outs"));

    c.def("calc_range_repeat_angles",
          [](T& s, f_in ins, f_in angles, f_out outs) {
              s.numpy_calc_range_angles(const_cast<float*>(ins.data()),
                                        const_cast<float*>(angles.data()),
                                        outs.mutable_data(),
                                        static_cast<int>(ins.shape(0)),
                                        static_cast<int>(angles.shape(0)));
          },
          py::arg("ins"), py::arg("angles"), py::arg("outs"));

    c.def("eval_sensor_model",
          [](T& s, f_in obs, f_in ranges, d_out outs, int num_rays, int num_particles) {
              s.eval_sensor_model(const_cast<float*>(obs.data()),
                                  const_cast<float*>(ranges.data()),
                                  outs.mutable_data(), num_rays, num_particles);
          },
          py::arg("observation"), py::arg("ranges"), py::arg("outs"),
          py::arg("num_rays"), py::arg("num_particles"));

    c.def("set_sensor_model",
          [](T& s, d_in table) {
              if (table.ndim() != 2 || table.shape(0) != table.shape(1)) {
                  py::print("Sensor model must have equal matrix dimensions, failing!");
                  return;
              }
              s.set_sensor_model(const_cast<double*>(table.data()),
                                 static_cast<int>(table.shape(0)));
          },
          py::arg("table"));

    c.def("calc_range_repeat_angles_eval_sensor_model",
          [](T& s, f_in ins, f_in angles, f_in obs, d_out weights) {
              s.calc_range_repeat_angles_eval_sensor_model(
                  const_cast<float*>(ins.data()), const_cast<float*>(angles.data()),
                  const_cast<float*>(obs.data()), weights.mutable_data(),
                  static_cast<int>(ins.shape(0)), static_cast<int>(angles.shape(0)));
          },
          py::arg("ins"), py::arg("angles"), py::arg("obs"), py::arg("weights"));

    if (has_save_trace)
        c.def("saveTrace", [](T& s, std::string p) { s.saveTrace(p); }, py::arg("path"));
}

PYBIND11_MODULE(range_libc, m) {
    m.doc() = "range_libc — pybind11 binding (header-only, no Cython/numpy at build)";

    // module flags (parity with the old wrapper)
    m.attr("USE_CACHED_TRIG")     = static_cast<bool>(_USE_CACHED_TRIG);
    m.attr("USE_ALTERNATE_MOD")   = static_cast<bool>(_USE_ALTERNATE_MOD);
    m.attr("USE_CACHED_CONSTANTS")= static_cast<bool>(_USE_CACHED_CONSTANTS);
    m.attr("USE_FAST_ROUND")      = static_cast<bool>(_USE_FAST_ROUND);
    m.attr("NO_INLINE")           = static_cast<bool>(_NO_INLINE);
    m.attr("USE_LRU_CACHE")       = static_cast<bool>(_USE_LRU_CACHE);
    m.attr("SHOULD_USE_CUDA")     = static_cast<bool>(USE_CUDA);

    py::class_<OMap>(m, "PyOMap")
        .def(py::init([](py::object a1, py::object a2, float resolution,
                         float origin_x, float origin_y, float origin_theta) {
                 return make_omap(a1, a2, resolution, origin_x, origin_y, origin_theta);
             }),
             py::arg("arg1"), py::arg("arg2") = py::none(),
             py::arg("resolution") = 1.0f, py::arg("origin_x") = 0.0f,
             py::arg("origin_y") = 0.0f, py::arg("origin_theta") = 0.0f)
        .def("save", [](OMap& s, std::string fn) { return s.save(fn); }, py::arg("filename"))
        .def("isOccupied", [](OMap& s, int x, int y) { return s.get(x, y); })
        .def("error", [](OMap& s) { return s.error(); })
        .def("width", [](OMap& s) { return static_cast<int>(s.width); })
        .def("height", [](OMap& s) { return static_cast<int>(s.height); });

    {
        py::class_<BresenhamsLine> c(m, "PyBresenhamsLine");
        c.def(py::init([](OMap& mp, float mr) { return new BresenhamsLine(mp, mr); }),
              py::arg("omap"), py::arg("max_range"));
        bind_methods<BresenhamsLine>(c, true, true);
    }
    {
        py::class_<RayMarching> c(m, "PyRayMarching");
        c.def(py::init([](OMap& mp, float mr) { return new RayMarching(mp, mr); }),
              py::arg("omap"), py::arg("max_range"));
        bind_methods<RayMarching>(c, true, true);
    }
    {
        py::class_<GiantLUTCast> c(m, "PyGiantLUTCast");
        c.def(py::init([](OMap& mp, float mr, unsigned int td) {
                  return new GiantLUTCast(mp, mr, td);
              }),
              py::arg("omap"), py::arg("max_range"), py::arg("theta_disc"));
        bind_methods<GiantLUTCast>(c, true, false);
    }
    {
        py::class_<CDDTCastPy> c(m, "PyCDDTCast");
        c.def(py::init([](OMap& mp, float mr, unsigned int td) {
                  return new CDDTCastPy(mp, mr, td);
              }),
              py::arg("omap"), py::arg("max_range"), py::arg("theta_disc"));
        bind_methods<CDDTCastPy>(c, true, false);
        c.def("prune",
              [](CDDTCastPy& s, float max_range) {
                  s.prune(max_range < 0.0f ? s.py_max_range : max_range);
              },
              py::arg("max_range") = -1.0f);
        c.def("calc_range_many_radial_optimized",
              [](CDDTCastPy& s, int num_rays, float min_angle, float max_angle,
                 f_in ins, f_out outs) {
                  s.calc_range_many_radial_optimized(
                      const_cast<float*>(ins.data()), outs.mutable_data(),
                      static_cast<int>(ins.shape(0)), num_rays, min_angle, max_angle);
              },
              py::arg("num_rays"), py::arg("min_angle"), py::arg("max_angle"),
              py::arg("ins"), py::arg("outs"));
    }

    // GPU backend: only real under a CUDA build. Without CUDA, expose a stub that
    // warns on construction (matches the old SHOULD_USE_CUDA==False behavior) so
    // `range_libc.PyRayMarchingGPU` still imports.
    {
        struct GPUStub {};
        py::class_<GPUStub> c(m, "PyRayMarchingGPU");
        c.def(py::init([](py::object, float) {
                  py::print("CANNOT USE RayMarchingGPU - compile RangeLib with USE_CUDA=1");
                  return new GPUStub();
              }),
              py::arg("omap"), py::arg("max_range"));
        c.def("calc_range_many", [](GPUStub&, f_in, f_out) {});
        c.def("calc_range_repeat_angles", [](GPUStub&, f_in, f_in, f_out) {});
        c.def("eval_sensor_model", [](GPUStub&, f_in, f_in, d_out, int, int) {});
        c.def("set_sensor_model", [](GPUStub&, d_in) {});
        c.def("calc_range_repeat_angles_eval_sensor_model",
              [](GPUStub&, f_in, f_in, f_in, d_out) {});
    }
}

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <solvers/RK4.h>

namespace py = pybind11;

template<unsigned short N>
void bind_RK4(py::module_ m) {
    py::class_<RK4<N>>(m, ("RK4_" + std::to_string(N) + "dim").c_str(), ("An RK4 solver for 1st order differential equations with " + std::to_string(N) + "-dimensional variable space").c_str())
        .def(py::init<std::function<std::array<double, N>(std::array<double, N>, double)>, std::array<double, N>, double>(),
            py::arg("f"), py::arg("y0"), py::arg("t0"),
            "initialize the solver with the function giving y'(y,t) and starting values and time")
        .def("current_time", &RK4<N>::GetCurrentTime)
        .def("current_value", &RK4<N>::GetCurrentValue)
        .def("times", &RK4<N>::GetTimes)
        .def("values", &RK4<N>::GetValues)
        .def("step", &RK4<N>::MakeStep,
            py::arg("dt"), "have the solver make a step of width dt")
        .def("steps", &RK4<N>::MakeSteps,
            py::arg("dt"), py::arg("K"), "have the solver make K steps of width dt")
        .def("steps_vector", &RK4<N>::MakeStepsVector,
            py::arg("dT"), "have the solver make steps for the widths in dT");
}

PYBIND11_MODULE(cosmos, module) {
    module.doc() = "My personal collection of C++ modules for cosmology";

    // submodule for numerical solvers
    py::module_ solvers = module.def_submodule("solvers", "A library of RK4 solvers for differential equations of different orders");

    bind_RK4<1>(solvers);
    bind_RK4<2>(solvers);
    bind_RK4<3>(solvers);
    bind_RK4<4>(solvers);
    bind_RK4<5>(solvers);
};



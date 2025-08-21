#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <solvers/RK4_1st.h>

namespace py = pybind11;

PYBIND11_MODULE(cosmos, module) {
    module.doc() = "My personal collection of C++ modules for cosmology";

    // submodule for numerical solvers
    py::module_ solvers = module.def_submodule("solvers", "A library of RK4 solvers for differential equations of different orders");

    py::class_<RK4_1st>(solvers, "RK4_1st", "RK4 solver for 1st order differential equations")
        .def(py::init<std::function<double(double, double)>, double, double>(),
            py::arg("f"), py::arg("y0"), py::arg("t0"),
            "initialize the solver with the function giving y'(y,t) and starting value and time")
        .def("current_time", &RK4_1st::GetCurrentTime)
        .def("current_value", &RK4_1st::GetCurrentValue)
        .def("times", &RK4_1st::GetTimes)
        .def("values", &RK4_1st::GetValues)
        .def("step", &RK4_1st::MakeStep,
            py::arg("dt"), "have the solver make a step of width dt")
        .def("steps", &RK4_1st::MakeSteps,
            py::arg("dt"), py::arg("N"), "have the solver make N steps of width dt")
        .def("steps_vector", &RK4_1st::MakeStepsVector,
            py::arg("dT"), "have the solver make steps for the widths in dT");

};



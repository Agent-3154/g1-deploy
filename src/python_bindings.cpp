#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "robot_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(g1_interface, m) {
    m.doc() = "Python bindings for G1Interface";

    // Expose RobotData struct
    py::class_<RobotData>(m, "RobotData")
        .def_readonly("q", &RobotData::q)
        .def_readonly("dq", &RobotData::dq)
        .def_readonly("tau", &RobotData::tau)
        .def_readonly("quaternion", &RobotData::quaternion)
        .def_readonly("rpy", &RobotData::rpy)
        .def_readonly("omega", &RobotData::omega);

    // Expose G1Interface class
    py::class_<G1Interface>(m, "G1Interface")
        .def(py::init<std::string>(), py::arg("networkInterface"))
        .def("load_mjcf", &G1Interface::loadMJCF, py::arg("mjcf_path"), "Load the MuJoCo model")
        .def("get_data", &G1Interface::getData, "Get the current robot data");
}


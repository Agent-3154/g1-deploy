#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "robot_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(g1_interface, m) {
    m.doc() = "Python bindings for G1Interface";

    // Expose RobotData<float> for hardware interface
    py::class_<RobotData<float>>(m, "RobotDataFloat")
        .def_readonly("q", &RobotData<float>::q)
        .def_readonly("dq", &RobotData<float>::dq)
        .def_readonly("tau", &RobotData<float>::tau)
        .def_readonly("quaternion", &RobotData<float>::quaternion)
        .def_readonly("rpy", &RobotData<float>::rpy)
        .def_readonly("omega", &RobotData<float>::omega);

    // Expose RobotData<double> as RobotData (Python's default is float64/double)
    py::class_<RobotData<double>>(m, "RobotData")
        .def_readonly("q", &RobotData<double>::q)
        .def_readonly("dq", &RobotData<double>::dq)
        .def_readonly("tau", &RobotData<double>::tau)
        .def_readonly("quaternion", &RobotData<double>::quaternion)
        .def_readonly("rpy", &RobotData<double>::rpy)
        .def_readonly("omega", &RobotData<double>::omega);

    // Expose G1HarwareInterface (uses float)
    py::class_<G1HarwareInterface>(m, "G1HarwareInterface")
        .def(py::init<std::string>(), py::arg("networkInterface"))
        .def("get_data", &G1HarwareInterface::getData, "Get the current robot data");

    // Expose G1MujocoInterface (uses double)
    py::class_<G1MujocoInterface>(m, "G1MujocoInterface")
        .def(py::init<std::string, double>(), 
             py::arg("mjcf_path"), 
             py::arg("timestep") = -1.0,
             "Initialize MuJoCo interface")
        .def("load_mjcf", &G1MujocoInterface::loadMJCF, py::arg("mjcf_path"), "Load the MuJoCo model")
        .def("get_data", &G1MujocoInterface::getData, "Get the current robot data")
        .def("run_async", &G1MujocoInterface::run_async, "Start asynchronous physics simulation")
        .def("stop", &G1MujocoInterface::stop, "Stop asynchronous physics simulation")
        .def("is_running", &G1MujocoInterface::is_running, "Check if physics simulation is running")
        .def("set_timestep", &G1MujocoInterface::set_timestep, py::arg("timestep"), "Set physics timestep")
        .def("get_timestep", &G1MujocoInterface::get_timestep, "Get physics timestep")
        .def("reset", &G1MujocoInterface::reset, "Reset the simulation");
}


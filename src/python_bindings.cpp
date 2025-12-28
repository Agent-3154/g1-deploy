#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include "robot_interface.hpp"

namespace py = pybind11;
using namespace unitree::common;

PYBIND11_MODULE(g1_interface, m) {
    m.doc() = "Python bindings for G1Interface";

    // Expose GamepadState::ButtonState
    py::class_<GamepadState::ButtonState>(m, "ButtonState")
        .def_readonly("pressed", &GamepadState::ButtonState::pressed)
        .def_readonly("on_press", &GamepadState::ButtonState::on_press)
        .def_readonly("on_release", &GamepadState::ButtonState::on_release);

    // Expose GamepadState
    py::class_<GamepadState>(m, "GamepadState")
        .def_readonly("lx", &GamepadState::lx)
        .def_readonly("rx", &GamepadState::rx)
        .def_readonly("ry", &GamepadState::ry)
        .def_readonly("ly", &GamepadState::ly)
        .def_readonly("l2", &GamepadState::l2)
        .def_readonly("R1", &GamepadState::R1)
        .def_readonly("L1", &GamepadState::L1)
        .def_readonly("start", &GamepadState::start)
        .def_readonly("select", &GamepadState::select)
        .def_readonly("R2", &GamepadState::R2)
        .def_readonly("L2", &GamepadState::L2)
        .def_readonly("F1", &GamepadState::F1)
        .def_readonly("F2", &GamepadState::F2)
        .def_readonly("A", &GamepadState::A)
        .def_readonly("B", &GamepadState::B)
        .def_readonly("X", &GamepadState::X)
        .def_readonly("Y", &GamepadState::Y)
        .def_readonly("up", &GamepadState::up)
        .def_readonly("right", &GamepadState::right)
        .def_readonly("down", &GamepadState::down)
        .def_readonly("left", &GamepadState::left);

    // Expose RobotData<float> for hardware interface
    py::class_<RobotData<float>>(m, "RobotDataFloat")
        .def_readonly("root_pos_w", &RobotData<float>::root_pos_w)
        .def_readonly("root_lin_vel_w", &RobotData<float>::root_lin_vel_w)
        .def_readonly("root_ang_vel_w", &RobotData<float>::root_ang_vel_w)
        .def_readonly("q", &RobotData<float>::q)
        .def_readonly("dq", &RobotData<float>::dq)
        .def_readonly("tau", &RobotData<float>::tau)
        .def_readonly("quaternion", &RobotData<float>::quaternion)
        .def_readonly("projected_gravity", &RobotData<float>::projected_gravity)
        .def_readonly("rpy", &RobotData<float>::rpy)
        .def_readonly("omega", &RobotData<float>::omega)
        .def_readonly("body_positions", &RobotData<float>::body_positions)
        .def_readonly("body_quaternions", &RobotData<float>::body_quaternions);

    // Expose RobotData<double> as RobotData (Python's default is float64/double)
    py::class_<RobotData<double>>(m, "RobotData")
        .def_readonly("root_pos_w", &RobotData<double>::root_pos_w)
        .def_readonly("root_lin_vel_w", &RobotData<double>::root_lin_vel_w)
        .def_readonly("root_ang_vel_w", &RobotData<double>::root_ang_vel_w)
        .def_readonly("q", &RobotData<double>::q)
        .def_readonly("dq", &RobotData<double>::dq)
        .def_readonly("tau", &RobotData<double>::tau)
        .def_readonly("quaternion", &RobotData<double>::quaternion)
        .def_readonly("projected_gravity", &RobotData<double>::projected_gravity)
        .def_readonly("rpy", &RobotData<double>::rpy)
        .def_readonly("omega", &RobotData<double>::omega)
        .def_readonly("body_positions", &RobotData<double>::body_positions)
        .def_readonly("body_quaternions", &RobotData<double>::body_quaternions);

    // Helper function to convert numpy array or list to std::array<float, 29>
    // py::array_t<float> automatically accepts both numpy arrays and lists/tuples
    auto convert_to_float_array = [](py::array_t<float> arr) -> std::array<float, 29> {
        py::buffer_info buf = arr.request();
        if (buf.size != 29) {
            throw std::runtime_error("Array must have exactly 29 elements");
        }
        std::array<float, 29> result;
        float* ptr = static_cast<float*>(buf.ptr);
        std::copy(ptr, ptr + 29, result.begin());
        return result;
    };

    // Helper function to convert numpy array or list to std::array<double, 29>
    // py::array_t<double> automatically accepts both numpy arrays and lists/tuples
    auto convert_to_double_array = [](py::array_t<double> arr) -> std::array<double, 29> {
        py::buffer_info buf = arr.request();
        if (buf.size != 29) {
            throw std::runtime_error("Array must have exactly 29 elements");
        }
        std::array<double, 29> result;
        double* ptr = static_cast<double*>(buf.ptr);
        std::copy(ptr, ptr + 29, result.begin());
        return result;
    };

    // Expose G1HarwareInterface (uses float)
    py::class_<G1HarwareInterface>(m, "G1HarwareInterface")
        .def(py::init<std::string>(), py::arg("networkInterface"))
        .def("load_mjcf", &G1HarwareInterface::loadMJCF, py::arg("mjcf_path"), "Load the MuJoCo model")
        .def("get_data", &G1HarwareInterface::getData, "Get the current robot data")
        .def("set_joint_stiffness", [&convert_to_float_array](G1HarwareInterface& self, py::array_t<float> joint_stiffness) {
            self.setJointStiffness(convert_to_float_array(joint_stiffness));
        }, py::arg("joint_stiffness"), "Set the joint stiffness (accepts list, tuple, or numpy array)")
        .def("set_joint_damping", [&convert_to_float_array](G1HarwareInterface& self, py::array_t<float> joint_damping) {
            self.setJointDamping(convert_to_float_array(joint_damping));
        }, py::arg("joint_damping"), "Set the joint damping (accepts list, tuple, or numpy array)")
        .def("write_joint_position_target", [&convert_to_float_array](G1HarwareInterface& self, py::array_t<float> joint_position_target) {
            self.writeJointPositionTarget(convert_to_float_array(joint_position_target));
        }, py::arg("joint_position_target"), "Write the joint position target (accepts list, tuple, or numpy array)")
        .def("write_joint_velocity_target", [&convert_to_float_array](G1HarwareInterface& self, py::array_t<float> joint_velocity_target) {
            self.writeJointVelocityTarget(convert_to_float_array(joint_velocity_target));
        }, py::arg("joint_velocity_target"), "Write the joint velocity target (accepts list, tuple, or numpy array)")
        .def("get_gamepad", &G1HarwareInterface::getGamepadState, "Get the current gamepad state");

    // Expose G1MujocoInterface (uses double)
    py::class_<G1MujocoInterface>(m, "G1MujocoInterface")
        .def(py::init<std::string, double>(), 
             py::arg("mjcf_path"), 
             py::arg("timestep") = -1.0,
             "Initialize MuJoCo interface")
        .def("load_mjcf", &G1MujocoInterface::loadMJCF, py::arg("mjcf_path"), "Load the MuJoCo model")
        .def("get_data", &G1MujocoInterface::getData, "Get the current robot data")
        .def("set_joint_stiffness", [&convert_to_double_array](G1MujocoInterface& self, py::array_t<double> joint_stiffness) {
            self.setJointStiffness(convert_to_double_array(joint_stiffness));
        }, py::arg("joint_stiffness"), "Set the joint stiffness (accepts list, tuple, or numpy array)")
        .def("set_joint_damping", [&convert_to_double_array](G1MujocoInterface& self, py::array_t<double> joint_damping) {
            self.setJointDamping(convert_to_double_array(joint_damping));
        }, py::arg("joint_damping"), "Set the joint damping (accepts list, tuple, or numpy array)")
        .def("run_async", &G1MujocoInterface::run_async, "Start asynchronous physics simulation")
        .def("stop", &G1MujocoInterface::stop, "Stop asynchronous physics simulation")
        .def("is_running", &G1MujocoInterface::is_running, "Check if physics simulation is running")
        .def("set_timestep", &G1MujocoInterface::set_timestep, py::arg("timestep"), "Set physics timestep")
        .def("get_timestep", &G1MujocoInterface::get_timestep, "Get physics timestep")
        .def("reset", [&convert_to_double_array](G1MujocoInterface& self, py::array_t<double> joint_pos) {
            self.reset(convert_to_double_array(joint_pos));
        }, py::arg("joint_pos"), "Reset the simulation")
        .def("write_joint_position_target", [&convert_to_double_array](G1MujocoInterface& self, py::array_t<double> joint_position_target) {
            self.writeJointPositionTarget(convert_to_double_array(joint_position_target));
        }, py::arg("joint_position_target"), "Write the joint position target (accepts list, tuple, or numpy array)")
        .def("write_joint_velocity_target", [&convert_to_double_array](G1MujocoInterface& self, py::array_t<double> joint_velocity_target) {
            self.writeJointVelocityTarget(convert_to_double_array(joint_velocity_target));
        }, py::arg("joint_velocity_target"), "Write the joint velocity target (accepts list, tuple, or numpy array)");
}


cmake_minimum_required(VERSION 3.16...3.27)
project(g1-deploy
  VERSION 1.0.0
  LANGUAGES CXX)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Dynamically detect pybind11_DIR using pybind11-config
execute_process(
    COMMAND pybind11-config --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    message(STATUS "Found pybind11 cmake dir: ${pybind11_DIR}")
else()
    message(WARNING "Could not find pybind11-config, falling back to default search")
endif()

find_package(pybind11 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
# Try to find unitree_sdk2 as a CMake package first (if installed via make install)
find_package(unitree_sdk2 REQUIRED)

# Find Mujoco from pip installation
# 1) Discover Python & ensure we can run a small snippet
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# 2) Ask Python where the installed 'mujoco' package lives
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
  OUTPUT_VARIABLE MUJOCO_PY_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mujoco, os; print(mujoco.__version__)"
  OUTPUT_VARIABLE MUJOCO_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 3) Derive lib & include (if present) paths from the package
set(MUJOCO_LIB    "${MUJOCO_PY_DIR}/libmujoco.so.${MUJOCO_VERSION}")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_PY_DIR}/include")   # may exist depending on wheel

message(STATUS "MuJoCo (pip) package dir: ${MUJOCO_PY_DIR}")
message(STATUS "MuJoCo lib:           ${MUJOCO_LIB}")
message(STATUS "MuJoCo include dir:       ${MUJOCO_INCLUDE_DIR}")

# Include FetchContent for downloading dependencies
include(FetchContent)
# Fetch Rerun SDK
FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

# Generate a compile_commands.json file for tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add executable
add_executable(g1-deploy
  cpp/main.cpp
)

# Include directories
target_include_directories(g1-deploy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MUJOCO_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(g1-deploy PRIVATE
    unitree_sdk2
    ${MUJOCO_LIB}
    Threads::Threads
    Eigen3::Eigen
    dl
)

target_link_libraries(g1-deploy PRIVATE rerun_sdk)

# Python module
# Note: Module name is _cpp to match the import in g1_deploy/__init__.py
pybind11_add_module(_cpp cpp/python_bindings.cpp cpp/robot_interface.hpp)

target_include_directories(_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(_cpp PRIVATE
    unitree_sdk2
    ${MUJOCO_LIB}
    Threads::Threads
    Eigen3::Eigen
    dl
)

install(TARGETS _cpp
    LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}"
    RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}"
)


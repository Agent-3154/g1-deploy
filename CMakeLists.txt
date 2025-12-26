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

# Try to find unitree_sdk2 as a CMake package first (if installed via make install)
find_package(unitree_sdk2 REQUIRED)

# Include FetchContent for downloading dependencies
include(FetchContent)

set(MUJOCO_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mujoco-3.4.0")

if(EXISTS "${MUJOCO_LOCAL_PATH}/CMakeLists.txt")
    message(STATUS "Found MuJoCo in local thirdparty directory: ${MUJOCO_LOCAL_PATH}")
    
    # Set MuJoCo build options before adding subdirectory
    set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "Build MuJoCo examples")
    set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "Build MuJoCo tests")
    set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "Build MuJoCo simulate application")
    
    # Add MuJoCo as a subdirectory
    add_subdirectory(${MUJOCO_LOCAL_PATH} ${CMAKE_BINARY_DIR}/_deps/mujoco-build)
    
else()
    message(FATAL_ERROR "Could not find MuJoCo in local thirdparty directory: ${MUJOCO_LOCAL_PATH}")
endif()

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
  src/main.cpp
)

# Include directories
target_include_directories(g1-deploy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(g1-deploy PRIVATE
    unitree_sdk2
    mujoco::mujoco
    Threads::Threads
    dl
)

target_link_libraries(g1-deploy PRIVATE rerun_sdk)

# Python module
pybind11_add_module(g1_interface src/python_bindings.cpp src/robot_interface.hpp)

target_include_directories(g1_interface PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(g1_interface PRIVATE
    unitree_sdk2
    mujoco::mujoco
    Threads::Threads
    dl
)

# Compile definitions (if needed)
# target_compile_definitions(g1-deploy PRIVATE ...)


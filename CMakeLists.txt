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
    
    # Add MuJoCo as a subdirectory (MuJoCo will handle Eigen3 dependency)
    # Note: If you get an error about Eigen3::Eigen already existing, try:
    # rm -rf build && mkdir build && cd build && cmake ..
    add_subdirectory(${MUJOCO_LOCAL_PATH} ${CMAKE_BINARY_DIR}/_deps/mujoco-build)
    
else()
    message(FATAL_ERROR "Could not find MuJoCo in local thirdparty directory: ${MUJOCO_LOCAL_PATH}")
endif()

# Eigen3 will be provided by MuJoCo's dependencies
# Check if Eigen3::Eigen exists after MuJoCo is configured
if(NOT TARGET Eigen3::Eigen)
    # If MuJoCo didn't provide Eigen3, try to find it
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        message(WARNING "Eigen3 not found. MuJoCo should provide it via FetchContent.")
    endif()
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

# Link Eigen3 to executable as well (if available)
if(TARGET Eigen3::Eigen)
    target_link_libraries(g1-deploy PRIVATE Eigen3::Eigen)
endif()

target_link_libraries(g1-deploy PRIVATE rerun_sdk)

# Python module
pybind11_add_module(g1_interface src/python_bindings.cpp src/robot_interface.hpp)

target_include_directories(g1_interface PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen3 (provided by MuJoCo)
# Eigen3::Eigen should automatically provide include directories when linked
if(TARGET Eigen3::Eigen)
    target_link_libraries(g1_interface PRIVATE Eigen3::Eigen)
    # Verify include directories are set (they should be via INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(EIGEN3_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    if(EIGEN3_INCLUDES)
        message(STATUS "Eigen3::Eigen include directories: ${EIGEN3_INCLUDES}")
    else()
        # If not set, try to get from EIGEN3_INCLUDE_DIR variable
        if(DEFINED EIGEN3_INCLUDE_DIR)
            target_include_directories(g1_interface PRIVATE ${EIGEN3_INCLUDE_DIR})
            message(STATUS "Manually added Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
        endif()
    endif()
else()
    # Fallback: try to find Eigen3 and add include directories manually
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        target_include_directories(g1_interface PRIVATE ${EIGEN3_INCLUDE_DIR})
        message(STATUS "Using Eigen3 from find_package: ${EIGEN3_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Eigen3::Eigen target not found. Eigen3 should be provided by MuJoCo. Try: rm -rf build && mkdir build && cd build && cmake ..")
    endif()
endif()

target_link_libraries(g1_interface PRIVATE
    unitree_sdk2
    mujoco::mujoco
    Threads::Threads
    dl
)

# Compile definitions (if needed)
# target_compile_definitions(g1-deploy PRIVATE ...)


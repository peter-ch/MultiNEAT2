cmake_minimum_required(VERSION 3.10)
project(MultiNEAT)

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the project “src” directory for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Build the executable target using main.cpp ---
# Get all .cpp files in src directory
file(GLOB EXE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Exclude any file containing “bindings” (so main.cpp and the rest are included)
list(FILTER EXE_SOURCES EXCLUDE REGEX "src/.*Bindings.*\\.cpp")
# Add executable target
add_executable(multineat_exe ${EXE_SOURCES})
target_include_directories(multineat_exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_definitions(-UNDEBUG)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)

# --- Build the Python module target (shared library) using pybind11 ---
# For the Python module we want everything except main.cpp.
# First, get all .cpp files in src (recursively if needed)
file(GLOB_RECURSE PY_ALL "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Exclude main.cpp from the Python module; we also exclude any extra binding files from this group (we add them separately)
list(FILTER PY_ALL EXCLUDE REGEX "src/.*Main\\.cpp")
list(FILTER PY_ALL EXCLUDE REGEX "src/.*Bindings.*\\.cpp")
# Now specify the bindings file (assumed to be src/bindings.cpp) and combine with the rest.
set(PY_MODULE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp" ${PY_ALL})
# Use pybind11 to create a MODULE target called pymultineat. (Python will load a shared-library named ‘pymultineat’)
pybind11_add_module(pymultineat MODULE ${PY_MODULE_SOURCES})
include_directories(${Python_INCLUDE_DIRS})
target_include_directories(pymultineat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(pymultineat PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET ON
        VISIBILITY_INLINES_HIDDEN ON
    )
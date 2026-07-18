cmake_minimum_required(VERSION 3.15)

project(MultiNEAT VERSION 2.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(MULTINEAT_BUILD_DEMO "Build the C++ XOR demo executable" ON)
option(MULTINEAT_BUILD_PYTHON "Build the pymultineat Python extension" ON)
option(MULTINEAT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(MULTINEAT_CORE_SOURCES
    src/Genome.cpp
    src/Innovation.cpp
    src/NeuralNetwork.cpp
    src/Parameters.cpp
    src/PhenotypeBehavior.cpp
    src/Population.cpp
    src/Random.cpp
    src/Species.cpp
    src/Substrate.cpp
    src/Traits.cpp
    src/Utils.cpp
)

set(MULTINEAT_PUBLIC_HEADERS
    src/Assert.h
    src/Genes.h
    src/Genome.h
    src/Innovation.h
    src/MultiNEATAssert.h
    src/NeuralNetwork.h
    src/Parameters.h
    src/PhenotypeBehavior.h
    src/Population.h
    src/Random.h
    src/Species.h
    src/Substrate.h
    src/Traits.h
    src/Utils.h
)

add_library(multineat ${MULTINEAT_CORE_SOURCES})
add_library(MultiNEAT::multineat ALIAS multineat)
target_compile_features(multineat PUBLIC cxx_std_17)
set_target_properties(multineat PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(multineat
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/MultiNEAT>"
)

function(multineat_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(MULTINEAT_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(MULTINEAT_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

multineat_enable_warnings(multineat)

if(MULTINEAT_BUILD_DEMO)
    add_executable(multineat_exe src/Main.cpp)
    target_link_libraries(multineat_exe PRIVATE MultiNEAT::multineat)
    multineat_enable_warnings(multineat_exe)
endif()

if(MULTINEAT_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 QUIET CONFIG)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            RESULT_VARIABLE _pybind11_result
            OUTPUT_VARIABLE _pybind11_dir
            ERROR_VARIABLE _pybind11_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT _pybind11_result EQUAL 0)
            message(FATAL_ERROR
                "pybind11 was not found. Install it for ${Python3_EXECUTABLE}: "
                "${_pybind11_error}"
            )
        endif()
        find_package(pybind11 REQUIRED CONFIG PATHS "${_pybind11_dir}" NO_DEFAULT_PATH)
    endif()

    pybind11_add_module(pymultineat MODULE src/Bindings.cpp)
    target_link_libraries(pymultineat PRIVATE MultiNEAT::multineat)
    multineat_enable_warnings(pymultineat)
endif()

if(BUILD_TESTING)
    add_executable(multineat_tests tests/core_tests.cpp)
    target_link_libraries(multineat_tests PRIVATE MultiNEAT::multineat)
    multineat_enable_warnings(multineat_tests)
    add_test(NAME multineat_core COMMAND multineat_tests)

    if(MULTINEAT_BUILD_PYTHON)
        if(MSVC)
            # A multi-config MSVC Debug extension links against python_d and
            # requires a debug Python interpreter, which standard
            # installations do not provide. The C++ Debug suite remains
            # enabled.
            add_test(
                NAME multineat_python
                CONFIGURATIONS Release RelWithDebInfo MinSizeRel
                COMMAND
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py"
            )
        else()
            add_test(
                NAME multineat_python
                COMMAND
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py"
            )
        endif()
        set_tests_properties(multineat_python PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:pymultineat>"
        )
    endif()
endif()

install(
    TARGETS multineat
    EXPORT MultiNEATTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/MultiNEAT"
)
install(
    FILES ${MULTINEAT_PUBLIC_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/MultiNEAT"
)

if(MULTINEAT_BUILD_PYTHON)
    install(
        TARGETS pymultineat
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()

set(MULTINEAT_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/MultiNEAT"
    CACHE STRING "Directory for the MultiNEAT CMake package files"
)

configure_package_config_file(
    cmake/MultiNEATConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MultiNEATConfig.cmake"
    INSTALL_DESTINATION "${MULTINEAT_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MultiNEATConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(
    EXPORT MultiNEATTargets
    FILE MultiNEATTargets.cmake
    NAMESPACE MultiNEAT::
    DESTINATION "${MULTINEAT_INSTALL_CMAKEDIR}"
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/MultiNEATConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/MultiNEATConfigVersion.cmake"
    DESTINATION "${MULTINEAT_INSTALL_CMAKEDIR}"
)
install(
    FILES LICENSE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/MultiNEAT"
)

cmake_minimum_required(VERSION 3.0)
project(sensei)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include(CMakeDependentOption)

# Default to release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
    CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
#------------------------------------------------------------------------------
# Set default output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#------------------------------------------------------------------------------
# Disable this option to build the miniapps without using any of the
# infrastructures, including the Sensei adaptor/brige.
option(ENABLE_SENSEI "Enable Sensei infrastucture" ON)

#------------------------------------------------------------------------------
# Options to enable various miniapps.
option(ENABLE_PARALLEL3D "Enable Parallel3D miniapp" ON)
option(ENABLE_OSCILLATORS "Enable Oscillators miniapp" ON)

#------------------------------------------------------------------------------
# Since Generic arrays are not yet VTK master, we enable it explicitly.
cmake_dependent_option(VTK_HAS_GENERIC_ARRAYS "VTK build has Generic arrays" OFF
  "ENABLE_SENSEI" OFF)

#------------------------------------------------------------------------------
# Enable Catalyst and analysis using Catalyst.
cmake_dependent_option(
  ENABLE_CATALYST "Enable analysis methods that use Catalyst" OFF "ENABLE_SENSEI" OFF)

#------------------------------------------------------------------------------
# Enable ADIOS and analysis using Adios
cmake_dependent_option(
  ENABLE_ADIOS "Enable analysis methods that use ADIOS" OFF "ENABLE_SENSEI" OFF)

#------------------------------------------------------------------------------
option(ENABLE_VTK_XMLP "Enable use of vtk parallel I/O" OFF)

#------------------------------------------------------------------------------
include(mpi)
if(ENABLE_ADIOS)
  include(adios)
endif()

add_subdirectory(utils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(ENABLE_SENSEI)
  message(STATUS "Enabled: Sensei infrastructure.")
  add_subdirectory(sensei)
else()
  message(STATUS "Disabled: Sensei infrastructure.")
endif()

add_subdirectory(miniapps)
add_subdirectory(endpoints)

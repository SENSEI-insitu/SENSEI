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

# Options to enable various analysis.
option(ENABLE_HISTOGRAM "Enable histogram analysis" ON)
option(ENABLE_AUTOCORRELATION "Enable autocorrelation analysis" ON)

# Options to enable various frameworks.

#------------------------------------------------------------------------------
# Enable Catalyst and analysis using Catalyst.
cmake_dependent_option(
  ENABLE_CATALYST "Enable analysis methods that use Catalyst" ON "ENABLE_SENSEI" OFF)
cmake_dependent_option(
  ENABLE_CATALYST_SLICE "Enable Catalyst Slice analysis" ON "ENABLE_CATALYST" OFF)

#------------------------------------------------------------------------------
# Enable ADIOS and analysis using Adios
cmake_dependent_option(
  ENABLE_ADIOS "Enable analysis methods that use ADIOS" ON "ENABLE_SENSEI" OFF)

#------------------------------------------------------------------------------
include(mpi)

add_subdirectory(utils)

# Process subdirectories.
if(ENABLE_SENSEI)
  message(STATUS "Enabled: Sensei infrastructure.")
  add_subdirectory(core)
else()
  message(STATUS "Disabled: Sensei infrastructure.")
endif()

if(ENABLE_HISTOGRAM)
  message(STATUS "Enabled: Histogram analysis.")
  add_subdirectory(analysis/histogram)
else()
  message(STATUS "Disabled: Histogram analysis.")
endif()

if(ENABLE_AUTOCORRELATION)
  message(STATUS "Enabled: Autocorrelation analysis.")
  add_subdirectory(analysis/autocorrelation)
else()
  message(STATUS "Disabled: Autocorrelation analysis.")
endif()

if(ENABLE_PARALLEL3D)
  message(STATUS "Enabled: Parallel3D miniapp.")
  add_subdirectory(miniapps/parallel3d)
else()
  message(STATUS "Disabled: Parallel3D miniapp.")
endif()

if(ENABLE_OSCILLATORS)
  message(STATUS "Enabled: Oscillators miniapp.")
  add_subdirectory(miniapps/oscillators)
else()
  message(STATUS "Disabled: Histogram miniapp.")
endif()

if(ENABLE_CATALYST)
  message(STATUS "Enabled: Catalyst")
  add_subdirectory(infrastructures/catalyst)
else()
  message(STATUS "Disabled: Catalyst")
endif()

if(ENABLE_ADIOS)
  message(STATUS "Enabled: ADIOS")
  add_subdirectory(infrastructures/adios)
else()
  message(STATUS "Disabled: ADIOS")
endif()

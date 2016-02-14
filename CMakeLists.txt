cmake_minimum_required(VERSION 2.8.12)
project(sensei)
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

# Options to enable various infrastructures.


#------------------------------------------------------------------------------
# Since we need MPI left and right, this makes it easier to deal with MPI.
find_package(MPI REQUIRED)
function(mpi_link target)
  target_include_directories(${target}
    SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
  target_compile_definitions(${target}
    PRIVATE ${MPI_C_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  target_link_libraries(${target}
    PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  if(MPI_C_LINK_FLAGS OR MPI_CXX_LINK_FLAGS)
    set_target_properties(${target}
      PROPERTIES LINK_FLAGS ${MPI_C_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})
  endif()
endfunction()

#------------------------------------------------------------------------------
# Process subdirectories.
if(ENABLE_SENSEI)
  message(STATUS "Enabled: Sensei infrastructure.")
  add_subdirectory(core)
else()
  message(STATUS "Disabled: Sensei infrastructure.")
endif()

if(ENABLE_HISTOGRAM)
  message(STATUS "Enabled: Histogram analysis.")
  add_subdirectory(histogram)
else()
  message(STATUS "Disabled: Histogram analysis.")
endif()

if(ENABLE_PARALLEL3D)
  message(STATUS "Enabled: Parallel3D miniapp.")
  add_subdirectory(parallel3d)
else()
  message(STATUS "Disabled: Parallel3D miniapp.")
endif()

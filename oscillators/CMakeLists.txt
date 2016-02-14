project                     (Oscillator)
cmake_minimum_required      (VERSION 2.8.11)

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

set                         (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

# Threads
find_package                (Threads)

# MPI
find_package                (MPI REQUIRED)
set                         (mpi_libraries              ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
                             SYSTEM ${MPI_INCLUDE_PATH})

# Format
add_library                 (util    src/format.cc)

add_executable              (oscillator oscillator.cpp analysis.cpp)
target_link_libraries       (oscillator util ${mpi_libraries} ${CMAKE_THREAD_LIBS_INIT})

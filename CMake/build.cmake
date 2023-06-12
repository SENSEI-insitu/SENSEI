set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(BUILD_SHARED_LIBS OFF "Build shared libraries by default")
option(BUILD_STATIC_EXECS  OFF "Link executables statically")
if (BUILD_STATIC_EXECS)
  set(BUILD_SHARED_LIBS OFF FORCE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LINK_SEARCH_START_STATIC TRUE)
  set(LINK_SEARCH_END_STATIC TRUE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
    CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# add the requisite flags. CMake enthusiasts will tell you that this is "not
# the CMake way". However, CMake has spotty coverage, is inconsistent in
# mechanisms, and often it does not work. Nonetheless, one may override our
# settings here by specifying them on the command line.
#
# Issues:
# * CMake does not propagate -fvisibility=hidden during cuda
#   linking and this leads to a nasty runtime crash when stataic and shared
#   libraries containing cuda code are linked into a fat bin.
#   Filed a bug report w/ NVIDIA and this has been reported to be fixed (Q4 2022)
# * CMake does not handle nvc++ as the CUDA compiler. (Q1 2023)
# * On some systems CMake will use O2 for release builds rather than O3 and does not
#   enable processor specific optimizations.
# * CMake's FindOpenMP module currently does not detect OpenMP offload flags at
# all. There is a a CMake bug report about this. (Q1 2023)
#
# these issues are likely to be resolved over time. adjust the CMake
# minimum version and test before removing these.

# import OpenMP offload flag detection from hamr rather than duplicated it here.
include(utils/HAMR/cmake/hamr_omp_offload.cmake)

if (NOT CMAKE_CXX_FLAGS)
    set(tmp "-fPIC -std=c++17 -Wall -Wextra -fvisibility=hidden")

    # this was necessary on MacOS in early days of C++11. likely not needed anymore.
    #if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
    #    set(tmp "${tmp} -stdlib=libc++")
    #endif()

    if (SENSEI_NVHPC_CUDA)
        set(tmp "${tmp} -cuda -gpu=${SENSEI_CUDA_ARCHITECTURES}")
    endif()

    if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        set(tmp "${tmp} -O3 -march=native -mtune=native")
    endif()

    if (SENSEI_ENABLE_OPENMP)
        set(tmp_flags)

        get_offload_compile_flags( TARGET ${SENSEI_OPENMP_TARGET}
            ARCH ${SENSEI_OPENMP_ARCH} ADD_FLAGS ${SENSEI_OPENMP_FLAGS}
            RESULT tmp_flags)

        set(tmp "${tmp} ${tmp_flags}")

        set(tmp_loop "distribute parallel for")
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
            set(tmp_loop "loop")
        endif()

        set(SENSEI_OPENMP_LOOP ${tmp_loop} CACHE
            STRING "OpenMP looping construct to use for device off load")
    endif()

    set(CMAKE_CXX_FLAGS "${tmp}"
        CACHE STRING "SENSEI C++ compiler defaults"
        FORCE)
    string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
        CACHE STRING "SENSEI C++ compiler defaults"
        FORCE)
endif()

set(tmp)
if (SENSEI_ENABLE_OPENMP)
  get_offload_link_flags(TARGET ${SENSEI_OPENMP_TARGET}
    ARCH ${SENSEI_OPENMP_ARCH} ADD_FLAGS ${SENSEI_OPENMP_FLAGS}
    RESULT tmp)
endif()

set(SENSEI_OPENMP_LINK_FLAGS ${tmp}
  CACHE STRING "SENSEI linker flags for OpenMP")

# CUDA
if (NOT CMAKE_CUDA_FLAGS AND SENSEI_NVCC_CUDA)
    set(tmp "--default-stream per-thread --expt-relaxed-constexpr") # -ccbin=${CMAKE_CXX_COMPILER}")
    if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
        set(tmp "${tmp} -Xcompiler -Wall,-Wextra,-O3,-march=native,-mtune=native,-fvisibility=hidden")
    elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
        set(tmp "${tmp} -g -G -Xcompiler -Wall,-Wextra,-O0,-g,-fvisibility=hidden")
    endif()

    set(CMAKE_CUDA_FLAGS "${tmp}"
        CACHE STRING "SENSEI CUDA compiler defaults"
        FORCE)

    string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
    set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
        CACHE STRING "SENSEI CUDA compiler defaults"
        FORCE)
endif()


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(sensei_CMAKE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/sensei)

message(STATUS "SENSEI: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "SENSEI: BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
message(STATUS "SENSEI: BUILD_STATIC_EXECS=${BUILD_STATIC_EXECS}")
message(STATUS "SENSEI: CMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message(STATUS "SENSEI: CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "SENSEI: CMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "SENSEI: CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "SENSEI: CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
message(STATUS "SENSEI: CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "SENSEI: CMAKE_CUDA_FLAGS_RELEASE=${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "SENSEI: CMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "SENSEI: CMAKE_CUDA_FLAGS_DEBUG=${CMAKE_CUDA_FLAGS_DEBUG}")

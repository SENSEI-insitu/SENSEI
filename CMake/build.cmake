option(BUILD_SHARED_LIBS ON "Build shared libraries (default: ON)")
option(BUILD_STATIC_EXECS  OFF "Link executables statically.")

if (BUILD_STATIC_EXECS)
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LINK_SEARCH_START_STATIC TRUE)
  set(LINK_SEARCH_END_STATIC TRUE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (BUILD_STATIC_EXECS)
  string(APPEND CMAKE_CXX_FLAGS "-static -static-libgcc -static-libstdc++ -pthread -Wl,-Bstatic")
endif ()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
message(STATUS "BUILD_STATIC_EXECS=${BUILD_STATIC_EXECS}")

add_compile_options(
   "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall;-Wextra>"
   # Need to explicitly set optimization level to -O3 because some systems default to -O2
   "$<$<CONFIG:Release>:-O3>"
   "$<$<CXX_COMPILER_ID:AppleClang>:-stdlib=libc++>")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

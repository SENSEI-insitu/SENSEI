#------------------------------------------------------------------------------
set(sources
  oscillator.cpp
  analysis.cpp)

#------------------------------------------------------------------------------
# Build the util library.
add_library(util STATIC src/format.cc)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# Build the oscillator executable.
add_executable(oscillator ${sources})
target_compile_options(oscillator PRIVATE "-std=c++11")
target_include_directories(oscillator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link with MPI.
target_link_libraries(oscillator PRIVATE mpi)

# Threads
find_package(Threads)
target_link_libraries(oscillator PRIVATE util ${CMAKE_THREAD_LIBS_INIT})

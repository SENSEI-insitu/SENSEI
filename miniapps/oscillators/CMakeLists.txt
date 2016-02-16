#------------------------------------------------------------------------------
set(sources
  oscillator.cpp)

if(ENABLE_SENSEI)
  list(APPEND sources
    dataadaptor.cpp
    bridge.cpp)
endif()
#------------------------------------------------------------------------------
# Build the util library.
add_library(util STATIC src/format.cc)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# Build the oscillator executable.
add_executable(oscillator ${sources})
target_compile_options(oscillator PRIVATE "-std=c++11")

# Link with MPI.
target_link_libraries(oscillator PRIVATE mpi diy grid)

# Threads
find_package(Threads)
target_link_libraries(oscillator PRIVATE util ${CMAKE_THREAD_LIBS_INIT})

if(ENABLE_SENSEI)
  target_link_libraries(oscillator PRIVATE core)
endif()

if(ENABLE_HISTOGRAM)
  target_link_libraries(oscillator PRIVATE histogram)
endif()

if(ENABLE_AUTOCORRELATION)
  target_link_libraries(oscillator PRIVATE autocorrelation)
endif()

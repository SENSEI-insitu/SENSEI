set(sources mandelbrot.cpp simulation_data.cpp patch.cpp)
set(libs m mpi timer)

if (ENABLE_SENSEI)
  list(APPEND sources MandelbrotDataAdaptor.cpp)
  list(APPEND libs sensei)
endif()

add_executable(mandelbrot ${sources})
target_link_libraries(mandelbrot PRIVATE ${libs})

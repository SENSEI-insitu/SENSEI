#-----------------------------------------------------------------------
set (sources
  histogram.cpp
  histogram.h
  )

if(ENABLE_SENSEI)
  list(APPEND sources
    vtk_histogram.cxx
    vtk_histogram.h)
endif()

#-----------------------------------------------------------------------
add_library(histogram STATIC ${sources})
target_compile_definitions(histogram INTERFACE ENABLE_HISTOGRAM)
include(mpi)
mpi_link(histogram)

if(ENABLE_SENSEI)
  target_link_libraries(histogram PRIVATE core)
endif()

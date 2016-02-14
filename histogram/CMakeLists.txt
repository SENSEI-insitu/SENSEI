#-----------------------------------------------------------------------
set (sources
  histogram.cpp
  histogram.h
  )
#-----------------------------------------------------------------------
add_library(histogram STATIC ${sources})
target_compile_definitions(histogram INTERFACE ENABLE_HISTOGRAM)
mpi_link(histogram)

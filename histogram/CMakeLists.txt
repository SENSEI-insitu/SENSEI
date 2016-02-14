#-----------------------------------------------------------------------
set (sources
  histogram.cpp
  histogram.h
  )

if(ENABLE_SENSEI)
  list(APPEND sources
    vtk_histogram.cxx
    vtk_histogram.h
    HistogramAnalysisAdaptor.cxx
    HistogramAnalysisAdaptor.h
    )
endif()

#-----------------------------------------------------------------------
add_library(histogram STATIC ${sources})
target_compile_definitions(histogram INTERFACE ENABLE_HISTOGRAM)
target_include_directories(histogram INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include(mpi)
mpi_link(histogram)

if(ENABLE_SENSEI)
  target_link_libraries(histogram PRIVATE core)
endif()

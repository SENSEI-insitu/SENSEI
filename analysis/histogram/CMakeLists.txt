#-----------------------------------------------------------------------
set(sources
  HistogramAnalysisAdaptor.cxx
  HistogramAnalysisAdaptor.h)

#-----------------------------------------------------------------------
add_library(histogram STATIC ${sources})
target_include_directories(histogram INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(histogram PUBLIC mpi core)

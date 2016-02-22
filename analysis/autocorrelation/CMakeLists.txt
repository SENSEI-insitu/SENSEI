#------------------------------------------------------------------------------
if(ENABLE_SENSEI)
  set(sources AutocorrelationAnalysisAdaptor.cxx)
else()
  set(sources analysis.cpp)
endif()

add_library(autocorrelation STATIC ${sources})
target_compile_options(autocorrelation PRIVATE "-std=c++11")
target_include_directories(autocorrelation INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autocorrelation PUBLIC mpi PRIVATE diy grid)
if (ENABLE_SENSEI)
  target_link_libraries(autocorrelation PUBLIC core)
endif()

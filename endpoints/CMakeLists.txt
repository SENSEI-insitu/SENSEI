if (ENABLE_ADIOS)
  #------------------------------------------------------------------------------
  add_executable(ADIOSAnalysisEndPoint ADIOSAnalysisEndPoint.cxx)
  target_link_libraries(ADIOSAnalysisEndPoint PRIVATE opts mpi adios sensei)
endif()

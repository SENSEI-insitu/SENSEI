if(ENABLE_ADIOS)
  #------------------------------------------------------------------------------
  add_executable(ADIOSAnalysisEndPoint ADIOSAnalysisEndPoint.cxx)
  target_link_libraries(ADIOSAnalysisEndPoint PRIVATE opts mpi adios sensei timer)
  #------------------------------------------------------------------------------
endif()

if(ENABLE_VTK_XMLP)
  #------------------------------------------------------------------------------
  add_executable(PosthocIOEndPoint PosthocIOEndPoint.cxx)
  target_link_libraries(PosthocIOEndPoint PRIVATE opts mpi sensei timer)
  #------------------------------------------------------------------------------
endif()

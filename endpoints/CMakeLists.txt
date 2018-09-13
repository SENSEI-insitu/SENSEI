if(ENABLE_ADIOS)
  add_executable(ADIOSAnalysisEndPoint ADIOSAnalysisEndPoint.cxx)
  target_link_libraries(ADIOSAnalysisEndPoint PRIVATE opts sensei timer sADIOS sMPI)
endif()

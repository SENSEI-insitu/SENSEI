if(ENABLE_ADIOS1)
  add_executable(ADIOS1EndPoint ADIOS1EndPoint.cpp)
  target_link_libraries(ADIOS1EndPoint PRIVATE opts sensei timer sADIOS1 sMPI)

  install(TARGETS ADIOS1EndPoint
    RUNTIME DESTINATION bin)
endif()

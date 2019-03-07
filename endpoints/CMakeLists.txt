if(ENABLE_ADIOS1)
  add_executable(ADIOS1EndPoint ADIOS1EndPoint.cpp)
  target_link_libraries(ADIOS1EndPoint PRIVATE opts sensei timer sADIOS1 sMPI)

  install(TARGETS ADIOS1EndPoint
    RUNTIME DESTINATION bin)
endif()

add_executable(ConfigurableEndPoint ConfigurableEndPoint.cxx)
target_link_libraries(ConfigurableEndPoint PRIVATE opts sensei timer pugixml sMPI)

install(TARGETS ConfigurableEndPoint RUNTIME DESTINATION bin)

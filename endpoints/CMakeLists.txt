if(ENABLE_ADIOS1)
  add_executable(ADIOS1EndPoint ADIOS1EndPoint.cpp)
  target_link_libraries(ADIOS1EndPoint PRIVATE opts sensei timer sADIOS1 sMPI)

  install(TARGETS ADIOS1EndPoint
    RUNTIME DESTINATION bin)
endif()

if(ENABLE_HDF5)
  add_executable(HDF5EndPoint HDF5EndPoint.cpp)
  #target_link_libraries(HDF5EndPoint PRIVATE opts sensei timer sHDF5 sMPI)
  target_link_libraries(HDF5EndPoint PRIVATE opts sensei sHDF5 sMPI)	

  install(TARGETS HDF5EndPoint
    RUNTIME DESTINATION bin)
endif()

add_executable(ConfigurableEndPoint ConfigurableEndPoint.cxx)
target_link_libraries(ConfigurableEndPoint PRIVATE opts sensei timer pugixml sMPI)

install(TARGETS ConfigurableEndPoint RUNTIME DESTINATION bin)

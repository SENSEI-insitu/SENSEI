set(endPointLibs opts sensei sMPI)

if(ENABLE_ADIOS1)
  list(APPEND endPointLibs sADIOS1)
endif()

if(ENABLE_HDF5)
  list(APPEND endPointLibs sHDF5)
endif()

add_executable(SENSEIEndPoint SENSEIEndPoint.cpp)
target_link_libraries(SENSEIEndPoint PRIVATE ${endPointLibs})
install(TARGETS SENSEIEndPoint RUNTIME DESTINATION bin)

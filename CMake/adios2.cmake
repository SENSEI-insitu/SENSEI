if(ENABLE_ADIOS2)
  find_package(ADIOS2 REQUIRED)

  add_library(sADIOS2 INTERFACE)
  target_link_libraries(sADIOS2 INTERFACE adios2::adios2)
  install(TARGETS sADIOS2 EXPORT sADIOS2)
  install(EXPORT sADIOS2 DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

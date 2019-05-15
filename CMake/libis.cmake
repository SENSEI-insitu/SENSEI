if(ENABLE_LIBIS)
  find_package(libIS REQUIRED)

  add_library(sLIBIS INTERFACE)
  target_link_libraries(sLIBIS INTERFACE is_sim)
  target_include_directories(sLIBIS SYSTEM INTERFACE ${libIS_INCLUDE_DIRS})

  install(TARGETS sLIBIS EXPORT sLIBIS)
  install(EXPORT sLIBIS DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

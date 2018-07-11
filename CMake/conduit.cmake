if(ENABLE_CONDUIT)
  find_package(Conduit REQUIRED)

  add_library(sconduit INTERFACE)

  target_link_libraries(sconduit INTERFACE conduit conduit_relay conduit_blueprint)

  target_include_directories(sconduit SYSTEM INTERFACE ${CONDUIT_INCLUDE_DIRS})

  install(TARGETS sconduit EXPORT sconduit)
  install(EXPORT sconduit DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

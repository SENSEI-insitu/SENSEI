if(ENABLE_CONDUIT)
  find_package(Conduit REQUIRED NO_DEFAULT_PATH PATHS ${CONDUIT_DIR}/lib/cmake)

  add_library(sConduit INTERFACE)

  target_link_libraries(sConduit INTERFACE conduit::conduit_mpi)
  target_include_directories(sConduit SYSTEM INTERFACE ${CONDUIT_INCLUDE_DIRS})

  install(TARGETS sConduit EXPORT sConduit)
  install(EXPORT sConduit DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

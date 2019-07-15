if(ENABLE_ASCENT)
  find_package(Ascent REQUIRED)
  find_package(Conduit REQUIRED)
  find_package(VTKh REQUIRED)
  find_package(VTKm REQUIRED)

  add_library(sAscent INTERFACE)

  target_link_libraries(sAscent INTERFACE ascent_mpi)
  target_include_directories(sAscent SYSTEM INTERFACE ${ASCENT_INCLUDE_DIRS} ${CONDUIT_INCLUDE_DIRS})

  install(TARGETS sAscent EXPORT sAscent)
  install(EXPORT sAscent DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

if(ENABLE_ASCENT)
  find_package(Ascent REQUIRED NO_DEFAULT_PATH PATHS ${ASCENT_DIR}/lib/cmake)

  if(NOT ENABLE_CONDUIT)
    add_library(sConduit INTERFACE)
  endif()

  add_library(sAscent INTERFACE)

  target_link_libraries(sAscent INTERFACE ascent::ascent_mpi)
  target_include_directories(sAscent SYSTEM INTERFACE ${ASCENT_INCLUDE_DIRS})

  install(TARGETS sAscent EXPORT sAscent)
  install(EXPORT sAscent DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

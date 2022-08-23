if(ENABLE_ASCENT)
  if (ASCENT_DIR)
    find_package(Ascent REQUIRED NO_DEFAULT_PATH PATHS ${ASCENT_DIR}/lib/cmake)
  else ()
    find_package(Ascent REQUIRED)
  endif ()

  if(NOT ENABLE_CONDUIT)
    add_library(sConduit INTERFACE)
  endif()

  add_library(sAscent INTERFACE)

  target_link_libraries(sAscent INTERFACE ascent::ascent_mpi)
  target_include_directories(sAscent SYSTEM INTERFACE ${ASCENT_INCLUDE_DIRS})

  install(TARGETS sAscent EXPORT sAscent)
  install(EXPORT sAscent DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

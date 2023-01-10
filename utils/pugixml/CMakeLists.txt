if (SENSEI_USE_EXTERNAL_pugixml)
  find_package(pugixml REQUIRED COMPONENTS pugixml)

  configure_file("${CMAKE_SOURCE_DIR}/CMake/external.pugixml.cmake.in"
    "${CMAKE_BINARY_DIR}/${sensei_CMAKE_INSTALL_CMAKEDIR}/pugixml.cmake"
    @ONLY)

  install(
    FILES "${CMAKE_BINARY_DIR}/${sensei_CMAKE_INSTALL_CMAKEDIR}/pugixml.cmake"
    DESTINATION "${sensei_CMAKE_INSTALL_CMAKEDIR}")
else ()
  add_library(pugixml STATIC src/pugixml.cpp)

  set_property(TARGET pugixml PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(pugixml
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/pugixml>)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
      DESTINATION include/pugixml FILES_MATCHING PATTERN "*.hpp")

  install(TARGETS pugixml EXPORT pugixml
      INCLUDES DESTINATION include/pugixml ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(EXPORT pugixml DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
      EXPORT_LINK_INTERFACE_LIBRARIES)
endif ()

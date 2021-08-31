if (SENSEI_USE_EXTERNAL_pugixml)
  find_package(pugixml REQUIRED COMPONENTS pugixml)

  configure_file("${CMAKE_SOURCE_DIR}/CMake/external.pugixml.cmake.in"
    "${CMAKE_BINARY_DIR}/lib/cmake/pugixml.cmake"
    @ONLY)

  install(
    FILES "${CMAKE_BINARY_DIR}/lib/cmake/pugixml.cmake"
    DESTINATION "lib/cmake")
else ()
  add_library(pugixml STATIC src/pugixml.cpp)

  target_include_directories(pugixml
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/pugixml>)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
      DESTINATION include/pugixml FILES_MATCHING PATTERN "*.hpp")

  install(TARGETS pugixml EXPORT pugixml
      INCLUDES DESTINATION include/pugixml ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)

  install(EXPORT pugixml DESTINATION lib/cmake
      EXPORT_LINK_INTERFACE_LIBRARIES)
endif ()

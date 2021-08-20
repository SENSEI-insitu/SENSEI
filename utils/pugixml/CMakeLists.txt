if(SENSEI_USE_EXTERNAL_pugixml)
  find_package(pugixml REQUIRED COMPONENTS pugixml)

  add_library(pugixml INTERFACE)
  install(TARGETS pugixml EXPORT pugixml)
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
endif ()

install(EXPORT pugixml DESTINATION lib/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

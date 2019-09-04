if(ENABLE_OPTS)
  add_library(sOPTS INTERFACE)

  target_include_directories(sOPTS SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS sOPTS EXPORT sOPTS)
  install(EXPORT sOPTS DESTINATION lib/cmake)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils/opts" DESTINATION include)
endif()


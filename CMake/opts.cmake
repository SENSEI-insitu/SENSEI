if(ENABLE_OPTS)
  add_library(sOPTS INTERFACE)

  target_include_directories(sOPTS SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/opts/include>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS sOPTS EXPORT sOPTS)
  install(EXPORT sOPTS DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR})
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils/opts/include" DESTINATION include)
endif()


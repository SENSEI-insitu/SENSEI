if(ENABLE_DIY)
  add_library(sDIY INTERFACE)

  target_include_directories(sDIY SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS sDIY EXPORT sDIY)
  install(EXPORT sDIY DESTINATION lib/cmake)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils/diy"
    DESTINATION include)
else()
  add_library(sDIY INTERFACE)

  target_include_directories(sDIY SYSTEM INTERFACE
    $<BUILD_INTERFACE:${DIY_DIR}>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS sDIY EXPORT sDIY)
  install(EXPORT sDIY DESTINATION lib/cmake)
  install(DIRECTORY "${DIY_DIR}"
    DESTINATION include)
endif()


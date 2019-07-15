if(REPLACE_DIY)
  add_library(sDIY INTERFACE)
  target_include_directories(sDIY SYSTEM INTERFACE
    $<BUILD_INTERFACE:${DIY_DIR}>
    $<INSTALL_INTERFACE:${DIY_DIR}>)
else()
  add_library(sDIY INTERFACE)

  target_include_directories(sDIY SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS sDIY EXPORT sDIY)
  install(EXPORT sDIY DESTINATION lib/cmake)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils/diy"
    DESTINATION include)
endif()


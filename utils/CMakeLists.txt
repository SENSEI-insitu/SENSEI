add_library(sDIY INTERFACE)
target_include_directories(sDIY SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
install(TARGETS sDIY EXPORT sDIY)
install(EXPORT sDIY DESTINATION lib/cmake)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/diy"
  DESTINATION include)

add_library(opts INTERFACE)
target_include_directories(opts
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opts"
  DESTINATION include)

add_subdirectory(pugixml)
#add_subdirectory(ImageTester)

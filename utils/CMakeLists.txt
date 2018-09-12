add_library(diy INTERFACE)
target_include_directories(diy SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
install(TARGETS diy EXPORT diy)
install(EXPORT diy DESTINATION lib/cmake)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/diy"
  DESTINATION include)

add_library(opts INTERFACE)
target_include_directories(opts
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opts"
  DESTINATION include)

add_subdirectory(pugixml)
add_subdirectory(timer)
#add_subdirectory(ImageTester)

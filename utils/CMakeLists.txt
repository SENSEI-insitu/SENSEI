add_library(opts INTERFACE)
target_include_directories(opts
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opts"
  DESTINATION include)

add_subdirectory(pugixml)
#add_subdirectory(ImageTester)

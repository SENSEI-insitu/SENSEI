# DIY
add_library(sDIY INTERFACE)
target_include_directories(sDIY SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
install(TARGETS sDIY EXPORT sDIY)
install(EXPORT sDIY DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sdiy"
  DESTINATION include)

# opts
add_library(opts INTERFACE)
target_include_directories(opts
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opts"
  DESTINATION include)

# pugi
add_subdirectory(pugixml)

#add_subdirectory(ImageTester)

# compile and wrap VTK 9
include(SVTKconfig.cmake)
add_subdirectory(SVTK)
add_subdirectory(pysvtk)

# HAMR
include(HAMRconfig.cmake)
add_subdirectory(HAMR)

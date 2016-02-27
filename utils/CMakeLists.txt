add_library(diy INTERFACE)
target_include_directories(diy
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(grid INTERFACE)
target_include_directories(grid
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(opts INTERFACE)
target_include_directories(opts
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(pugixml STATIC
  pugixml/src/pugixml.cpp
  pugixml/src/pugixml.hpp
  pugixml/src/pugiconfig.hpp)
target_include_directories(pugixml
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src)

add_subdirectory(ArrayIO)

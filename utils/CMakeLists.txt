add_library(diy INTERFACE)
target_include_directories(diy
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(grid INTERFACE)
target_include_directories(grid
  SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

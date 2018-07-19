if (BUILD_TESTING)
  add_executable(CompareImages CompareImages.cxx)
  target_link_libraries(CompareImages vtk)
endif()

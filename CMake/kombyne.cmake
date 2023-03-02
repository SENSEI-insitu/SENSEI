if(ENABLE_KOMBYNE)
  cmake_policy(SET CMP0074 NEW)
  set(kombyne_ROOT "${KOMBYNE_DIR}")
  # gaatmp: should this be KOMBYNE_ENABLE_CUDA?
  if(ENABLE_CUDA)
    enable_language(CUDA)
  endif()
  find_package(kombyne)
  add_library(sKombyne INTERFACE)
  target_link_libraries(sKombyne INTERFACE kombyne)
  install(TARGETS sKombyne EXPORT sKombyne)
  install(EXPORT sKombyne DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

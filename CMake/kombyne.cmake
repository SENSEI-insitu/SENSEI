if(ENABLE_KOMBYNE)
  cmake_policy(SET CMP0074 NEW)
  if(ENABLE_CUDA)
    enable_language(CUDA)
  endif()

  if(DEFINED KOMBYNE_DIR)
    set(kombyne_ROOT "${KOMBYNE_DIR}")
    find_package(kombyne)

    add_library(sKombyne INTERFACE)
    target_link_libraries(sKombyne INTERFACE kombyne)
    target_include_directories(sKombyne SYSTEM INTERFACE
      "${KB_INSTALL_INCLUDEDIR}/kombyne")

    install(TARGETS sKombyne EXPORT sKombyne)
    install(EXPORT sKombyne DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
      EXPORT_LINK_INTERFACE_LIBRARIES)

  elseif(DEFINED KOMBYNELITE_DIR)
    set(kombynelite_ROOT "${KOMBYNELITE_DIR}")
    find_package(kombynelite)

    add_library(sKombyne INTERFACE)
    target_link_libraries(sKombyne INTERFACE kombynelite)
    target_include_directories(sKombyne SYSTEM INTERFACE
      "${KB_INSTALL_INCLUDEDIR}/kombynelite")

    install(TARGETS sKombyne EXPORT sKombyne)
    install(EXPORT sKombyne DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
      EXPORT_LINK_INTERFACE_LIBRARIES)

  endif()

endif()

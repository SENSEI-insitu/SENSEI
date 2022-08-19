if(ENABLE_LIBSIM)
  find_package(LIBSIM REQUIRED)
  add_library(sLibsim INTERFACE)
  target_link_libraries(sLibsim INTERFACE ${LIBSIM_LIBRARIES})
  target_include_directories(sLibsim SYSTEM INTERFACE ${LIBSIM_INCLUDE_DIRS})
  install(TARGETS sLibsim EXPORT sLibsim)
  install(EXPORT sLibsim DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

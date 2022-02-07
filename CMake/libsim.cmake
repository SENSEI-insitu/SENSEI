if(ENABLE_LIBSIM)
  find_package(LIBSIM REQUIRED)
  add_library(sLibsim INTERFACE)
  target_link_libraries(sLibsim INTERFACE ${LIBSIM_LIBRARIES})
  target_include_directories(sLibsim SYSTEM INTERFACE ${LIBSIM_INCLUDE_DIRS})
  install(TARGETS sLibsim EXPORT sLibsim)
  install(EXPORT sLibsim DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

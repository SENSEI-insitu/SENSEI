if (ENABLE_CATALYST)

  set(SENSEI_PV_COMPONENTS Catalyst RemotingViews)
  if(ENABLE_CATALYST_PYTHON)
    list(APPEND SENSEI_PV_COMPONENTS PythonCatalyst)
  endif()

  # get the paraview version
  find_package(ParaView CONFIG QUIET)

  if(NOT ParaView_FOUND)
    message(STATUS ${ParaView_NOT_FOUND_MESSAGE})
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build "
      "(or install directory). Please set ParaView_DIR to point to directory "
      "containing `ParaViewConfig.cmake` or `paraview-config.cmake`.")
  endif()

  # this helps thre user know what we've tested. OK to expand at any point if
  # it has been tested.
  if (ParaView_VERSION VERSION_LESS "5.9" OR ParaView_VERSION VERSION_GREATER_EQUAL "5.11")
    message(FATAL_ERROR "This release of SENSEI requires ParaView 5.9 - 5.10")
  endif()

  # find the paraview libraries
  find_package(ParaView CONFIG OPTIONAL_COMPONENTS ${SENSEI_PV_COMPONENTS})

  # find VTK separately
  find_package(VTK CONFIG QUIET COMPONENTS ${SENSEI_VTK_COMPONENTS})
  add_library(sVTK INTERFACE)
  target_link_libraries(sVTK INTERFACE ${ParaView_LIBRARIES} ${VTK_LIBRARIES})
  install(TARGETS sVTK EXPORT sVTK)
  install(EXPORT sVTK DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)

endif()

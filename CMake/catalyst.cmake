if (ENABLE_CATALYST)
  set(sensei_pv_components_legacy vtkPVCatalyst vtkPVServerManagerRendering)
  set(sensei_pv_components_5_7 Catalyst ServerManagerRendering)
  set(sensei_pv_components_5_8 Catalyst RemotingViews)

  if(ENABLE_CATALYST_PYTHON)
    list(APPEND sensei_pv_components_legacy vtkPVPythonCatalyst)
    list(APPEND sensei_pv_components_5_7 PythonCatalyst)
    list(APPEND sensei_pv_components_5_8 PythonCatalyst)
  endif()

  find_package(ParaView CONFIG QUIET)
  if(NOT ParaView_FOUND)
    message(STATUS ${ParaView_NOT_FOUND_MESSAGE})
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build "
      "(or install directory). Please set ParaView_DIR to point to directory "
      "containing `ParaViewConfig.cmake` or `paraview-config.cmake`.")
  endif()
  if (ParaView_VERSION VERSION_LESS "5.7.0")
    set (SENSEI_PV_COMPONENTS ${sensei_pv_components_legacy} ${SENSEI_VTK_COMPONENTS})
  elseif (ParaView_VERSION VERSION_LESS "5.8.0")
    set (SENSEI_PV_COMPONENTS ${sensei_pv_components_5_7})
  else()
    set (SENSEI_PV_COMPONENTS ${sensei_pv_components_5_8})
  endif()
  find_package(ParaView CONFIG COMPONENTS ${SENSEI_PV_COMPONENTS})

  # avoid leaking these internal variables
  unset(sensei_pv_components_legacy)
  unset(sensei_pv_components_5_7)
  unset(sensei_pv_components_5_8)

  add_library(sVTK INTERFACE)
  if (ParaView_VERSION VERSION_LESS "5.7.0")
    target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
    target_include_directories(sVTK SYSTEM INTERFACE ${PARAVIEW_INCLUDE_DIRS})
    target_compile_definitions(sVTK INTERFACE ${VTK_DEFINITIONS})
  else()
    # find VTK separately
    find_package(VTK CONFIG QUIET COMPONENTS ${SENSEI_VTK_COMPONENTS})
    target_link_libraries(sVTK INTERFACE ${ParaView_LIBRARIES} ${VTK_LIBRARIES})
  endif()

  install(TARGETS sVTK EXPORT sVTK)
  install(EXPORT sVTK DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

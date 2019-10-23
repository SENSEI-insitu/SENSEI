if (ENABLE_CATALYST)
  set(sensei_pv_components_legacy vtkPVCatalyst vtkPVServerManagerRendering)
  set(sensei_pv_components_modern Catalyst ServerManagerRendering)

  if(ENABLE_CATALYST_PYTHON)
    list(APPEND sensei_pv_components_legacy vtkPVPythonCatalyst)
    list(APPEND sensei_pv_components_modern PythonCatalyst)
  endif()

  find_package(ParaView CONFIG QUIET)
  if(NOT ParaView_FOUND)
    message(STATUS ${ParaView_NOT_FOUND_MESSAGE})
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build "
      "(or install directory. Please set ParaView_DIR to point to directory "
      "containing `ParaViewConfig.cmake` or `paraview-config.cmake`.")
  endif()
  if (ParaView_VERSION VERSION_LESS "5.7.0")
    set (SENSEI_PV_COMPONENTS ${sensei_pv_components_legacy} ${SENSEI_VTK_COMPONENTS})
  else()
    set (SENSEI_PV_COMPONENTS ${sensei_pv_components_modern})
  endif()
  find_package(ParaView CONFIG COMPONENTS ${SENSEI_PV_COMPONENTS})
  if(NOT ParaView_FOUND)
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build "
      "(or install directory. Please set ParaView_DIR to point to directory "
      "containing `ParaViewConfig.cmake` or `paraview-config.cmake`.")
  endif()

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
  install(EXPORT sVTK DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

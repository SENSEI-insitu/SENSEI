if (ENABLE_CATALYST)
  set(SENSEI_PV_COMPONENTS vtkPVCatalyst vtkPVServerManagerRendering)
  if(ENABLE_CATALYST_PYTHON)
    list(APPEND SENSEI_PV_COMPONENTS vtkPVPythonCatalyst)
  endif()

  find_package(ParaView COMPONENTS ${SENSEI_VTK_COMPONENTS}
    ${SENSEI_PV_COMPONENTS})

  if(NOT ParaView_FOUND)
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build"
      "(or install directory. Please set ParaView_DIR to point to " "directory"
      "containing `ParaViewConfig.cmake`.")
  endif()

  add_library(sVTK INTERFACE)
  target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
  target_include_directories(sVTK SYSTEM INTERFACE ${PARAVIEW_INCLUDE_DIRS})
  target_compile_definitions(sVTK INTERFACE ${VTK_DEFINITIONS})

  install(TARGETS sVTK EXPORT sVTK)
  install(EXPORT sVTK DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

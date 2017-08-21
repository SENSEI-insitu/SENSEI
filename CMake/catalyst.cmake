if (ENABLE_CATALYST)
  set(SENSEI_PV_COMPONENTS vtkPVCatalyst vtkPVServerManagerRendering)
  if(ENABLE_CATALYST_PYTHON)
    list(APPEND SENSEI_PV_COMPONENTS vtkPVPythonCatalyst)
  endif()

  if(ENABLE_CINEMA)
    list(APPEND SENSEI_PV_COMPONENTS
      vtkicet
      vtkPVVTKExtensionsRendering
      vtkAcceleratorsVtkm
    )
  endif()

  find_package(ParaView COMPONENTS ${SENSEI_VTK_COMPONENTS}
    ${SENSEI_PV_COMPONENTS})

  if(NOT ParaView_FOUND)
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build"
      "(or install directory. Please set ParaView_DIR to point to " "directory"
      "containing `ParaViewConfig.cmake`.")
  endif()

  add_library(vtk INTERFACE)
  target_link_libraries(vtk INTERFACE ${VTK_LIBRARIES})
  target_include_directories(vtk SYSTEM INTERFACE ${PARAVIEW_INCLUDE_DIRS})
  target_compile_definitions(vtk INTERFACE ${VTK_DEFINITIONS})

  install(TARGETS vtk EXPORT vtk)
  install(EXPORT vtk DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

set(SENSEI_ENABLE_VTK_CORE OFF)
if (SENSEI_ENABLE_VTK_MPI OR SENSEI_ENABLE_VTK_IO OR SENSEI_ENABLE_VTK_MPI OR
  SENSEI_ENABLE_VTK_RENDERING OR SENSEI_ENABLE_VTK_ACCELERATORS OR
  SENSEI_ENABLE_VTK_FILTERS OR SENSEI_ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY)

  set(SENSEI_ENABLE_VTK_CORE ON)

  # lets build the list of modules for VTK pre-8.90 and post 8.90
  set(sensei_vtk_components_legacy)
  set(sensei_vtk_components_modern CommonDataModel)

  if (SENSEI_ENABLE_VTK_MPI)
    list(APPEND sensei_vtk_components_legacy vtkParallelMPI)
    list(APPEND sensei_vtk_components_modern ParallelMPI)
  endif()

  if (SENSEI_ENABLE_VTK_IO)
    list(APPEND sensei_vtk_components_legacy vtkIOXML vtkIOLegacy)
    list(APPEND sensei_vtk_components_modern IOXML IOLegacy)
    if (SENSEI_ENABLE_VTK_MPI)
      list(APPEND sensei_vtk_components_legacy vtkIOParallelXML)
      list(APPEND sensei_vtk_components_modern IOParallelXML)
    endif()
  endif()

  if (SENSEI_ENABLE_VTK_RENDERING)
    list(APPEND sensei_vtk_components_legacy vtkRenderingCore)
    list(APPEND sensei_vtk_components_modern RenderingCore)
    if (TARGET vtkRenderingOpenGL2)
      list(APPEND sensei_vtk_components_legacy vtkRenderingOpenGL2)
      list(APPEND sensei_vtk_components_modern RenderingOpenGL2)
    endif()
    if (TARGET vtkRenderingOpenGL)
      list(APPEND sensei_vtk_components_legacy vtkRenderingOpenGL)
      list(APPEND sensei_vtk_components_modern RenderingOpenGL)
    endif()
  endif()

  if (SENSEI_ENABLE_VTK_ACCELERATORS)
    list(APPEND sensei_vtk_components_legacy vtkAcceleratorsVTKm vtkIOLegacy
      vtkFiltersGeometry vtkImagingCore)
    list(APPEND sensei_vtk_components_modern AcceleratorsVTKm IOLegacy
      FiltersGeometry ImagingCore)
  endif()

  if (SENSEI_ENABLE_VTK_FILTERS)
    list(APPEND sensei_vtk_components_legacy vtkFiltersGeneral)
    list(APPEND sensei_vtk_components_modern FiltersGeneral)
  endif()

  if (SENSEI_ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY)
    list(APPEND sensei_vtk_components_legacy vtkFiltersParallelGeometry)
    list(APPEND sensei_vtk_components_modern FiltersParallelGeometry)
  endif()

  if (sensei_vtk_components_modern)
    if (NOT SENSEI_ENABLE_CATALYST)
      add_library(sVTK INTERFACE)

      find_package(VTK CONFIG REQUIRED)

      if (VTK_VERSION VERSION_LESS "8.90.0")
        set(SENSEI_VTK_COMPONENTS ${sensei_vtk_components_legacy})
      else()
        set(SENSEI_VTK_COMPONENTS ${sensei_vtk_components_modern})
      endif()

      # avoid leaking these internal variables
      unset(sensei_vtk_components_legacy)
      unset(sensei_vtk_components_modern)

      find_package(VTK CONFIG REQUIRED COMPONENTS ${SENSEI_VTK_COMPONENTS})

      if (VTK_VERSION VERSION_LESS "8.90.0")
        target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
        target_include_directories(sVTK SYSTEM INTERFACE ${VTK_INCLUDE_DIRS})
        target_compile_definitions(sVTK INTERFACE ${VTK_DEFINITIONS})
      else()
        target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
      endif()

      install(TARGETS sVTK EXPORT sVTK)
      install(EXPORT sVTK DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
        EXPORT_LINK_INTERFACE_LIBRARIES)
    endif()
  endif()
endif()

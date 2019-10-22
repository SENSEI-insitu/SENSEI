# lets build the list of modules for VTK pre-8.90 and post 8.90
set(sensei_vtk_components_legacy)
set(sensei_vtk_components_modern)


set(sensei_vtk_components_legacy vtkCommonDataModel)
set(sensei_vtk_components_modern CommonDataModel)

# note: this may be a bug. VTKUtils::WriteDomainDecomp requires
# `vtkUnstructuredGridWriter`. Not sure if that requirement is reasonable. It
# adds a required dependency to `VTK::IOLegacy` which may not be a good idea in
# the long run.
list(APPEND sensei_vtk_components_legacy vtkIOLegacy)
list(APPEND sensei_vtk_components_modern IOLegacy)

if (ENABLE_VTK_MPI)
  list(APPEND sensei_vtk_components_legacy vtkParallelMPI)
  list(APPEND sensei_vtk_components_modern ParallelMPI)
endif()
if (ENABLE_VTK_IO)
  list(APPEND sensei_vtk_components_legacy vtkIOXML vtkIOLegacy)
  list(APPEND sensei_vtk_components_modern IOXML IOLegacy)
  if (ENABLE_VTK_MPI)
    list(APPEND sensei_vtk_components_legacy vtkIOParallelXML)
    list(APPEND sensei_vtk_components_modern IOParallelXML)
  endif()
endif()
if (ENABLE_VTK_RENDERING)
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
if (ENABLE_VTK_ACCELERATORS)
  list(APPEND sensei_vtk_components_legacy vtkAcceleratorsVTKm vtkIOLegacy
    vtkFiltersGeometry vtkImagingCore)
  list(APPEND sensei_vtk_components_modern AcceleratorsVTKm IOLegacy
    FiltersGeometry ImagingCore)
endif()
if (ENABLE_VTK_FILTERS)
  list(APPEND sensei_vtk_components_legacy vtkFiltersGeneral)
  list(APPEND sensei_vtk_components_modern FiltersGeneral)
endif()
if (ENABLE_PYTHON)
  list(APPEND sensei_vtk_components_legacy vtkPython vtkWrappingPythonCore)
  list(APPEND sensei_vtk_components_modern Python WrappingPythonCore)
endif()

if (NOT ENABLE_CATALYST)
  add_library(sVTK INTERFACE)

  find_package(VTK CONFIG QUIET)
  if (NOT VTK_FOUND)
    message(FATAL_ERROR "VTK is required for Sensei core even when not using "
      "any infrastructures. Please set `VTK_DIR` to point to a directory "
      "containing `VTKConfig.cmake` or `vtk-config.cmake`.")
  endif()

  if (VTK_VERSION VERSION_LESS "8.90.0")
    set(SENSEI_VTK_COMPONENTS ${sensei_vtk_components_legacy})
  else()
    set(SENSEI_VTK_COMPONENTS ${sensei_vtk_components_modern})
  endif()

  find_package(VTK CONFIG QUIET COMPONENTS ${SENSEI_VTK_COMPONENTS})
  if (NOT VTK_FOUND)
    message(FATAL_ERROR "VTK (${SENSEI_VTK_COMPONENTS}) modules are required for "
      "Sensei core even when not using any infrastructures. Please set "
      "`VTK_DIR` to point to a directory containing `VTKConfig.cmake` or "
      "`vtk-config.cmake`.")
  endif()

  if (VTK_VERSION VERSION_LESS "8.90.0")
    target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
    target_include_directories(sVTK SYSTEM INTERFACE ${VTK_INCLUDE_DIRS})
    target_compile_definitions(sVTK INTERFACE ${VTK_DEFINITIONS})
  else()
    target_link_libraries(sVTK INTERFACE ${VTK_LIBRARIES})
  endif()

  install(TARGETS sVTK EXPORT sVTK)
  install(EXPORT sVTK DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)
endif()

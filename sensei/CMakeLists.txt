#-----------------------------------------------------------------------
set (sources
  AnalysisAdaptor.cxx
  AnalysisAdaptor.h
  Autocorrelation.cxx
  Autocorrelation.h
  ConfigurableAnalysis.cxx
  ConfigurableAnalysis.h
  DataAdaptor.cxx
  DataAdaptor.h
)

if(VTK_HAS_GENERIC_ARRAYS)
  list(APPEND sources
    Histogram.cxx
    Histogram.h)
endif()

if(ENABLE_CATALYST)
  list(APPEND sources
    catalyst/AnalysisAdaptor.cxx
    catalyst/AnalysisAdaptor.h
    catalyst/Slice.cxx
    catalyst/Slice.h)
endif()
if(ENABLE_ADIOS)
  list(APPEND sources
    adios/AnalysisAdaptor.cxx
    adios/AnalysisAdaptor.h
    adios/DataAdaptor.cxx
    adios/DataAdaptor.h)
endif()

#-----------------------------------------------------------------------
add_library(sensei STATIC ${sources})
target_compile_definitions(sensei INTERFACE ENABLE_SENSEI)

if(ENABLE_CATALYST)
  find_package(ParaView QUIET COMPONENTS vtkPVCatalyst vtkCommonDataModel)
  if(NOT ParaView_FOUND)
    message(FATAL_ERROR "Catalyst analysis components require Catalyst build (or "
                        "install directory. Please set ParaView_DIR to point to "
                        "directory containing `ParaViewConfig.cmake`.")
  endif()
  target_include_directories(sensei SYSTEM PUBLIC ${PARAVIEW_INCLUDE_DIRS})
  target_compile_definitions(sensei PUBLIC ${VTK_DEFINITIONS})
  target_link_libraries(sensei PUBLIC ${VTK_LIBRARIES})
  target_compile_definitions(sensei PUBLIC ENABLE_CATALYST)
else()
  find_package(VTK QUIET COMPONENTS vtkCommonDataModel)
  if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK (vtkCommonDataModel) is required for Sensei core "
      "even when not using any infrastructures. Please set VTK_DIR to point to a "
      "directory containing `VTKConfig.cmake`.")
  endif()
  target_include_directories(sensei SYSTEM PUBLIC ${VTK_INCLUDE_DIRS})
  target_compile_definitions(sensei PUBLIC ${VTK_DEFINITIONS})
endif()

if(VTK_HAS_GENERIC_ARRAYS)
  target_compile_definitions(sensei PRIVATE ENABLE_HISTOGRAM)
endif()

if(ENABLE_ADIOS)
  target_link_libraries(sensei PRIVATE adios)
  target_compile_definitions(sensei PUBLIC ENABLE_ADIOS)
endif()

target_link_libraries(sensei
  PUBLIC  ${VTK_LIBRARIES}
          mpi
  PRIVATE diy
          grid
          pugixml
          timer
  )

#------------------------------------------------------------------------------
# Needed by Autocorrelation.cxx
find_package(Threads)
target_link_libraries(sensei PUBLIC ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(sensei PUBLIC cxx_auto_type)

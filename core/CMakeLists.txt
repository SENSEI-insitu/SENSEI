#-----------------------------------------------------------------------
set (sources
  vtkInsituAnalysisAdaptor.cxx
  vtkInsituAnalysisAdaptor.h
  vtkInsituDataAdaptor.cxx
  vtkInsituDataAdaptor.h
  )

#-----------------------------------------------------------------------
find_package(VTK QUIET COMPONENTS vtkCommonDataModel)
if(NOT VTK_FOUND)
  message(FATAL_ERROR "VTK (vtkCommonDataModel) is required for Sensei core "
    "even when not using any infrastructures. Please set VTK_DIR to point to a "
    "directory containing `VTKConfig.cmake`.")
endif()

add_library(core STATIC ${sources})
target_include_directories(core SYSTEM PUBLIC ${VTK_INCLUDE_DIRS}
                                INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(core PUBLIC ${VTK_DEFINITIONS}
                                INTERFACE ENABLE_SENSEI)
target_link_libraries(core PUBLIC ${VTK_LIBRARIES})

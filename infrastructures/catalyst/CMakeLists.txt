set(sources
  vtkCatalystAnalysisAdaptor.cxx
  vtkCatalystSlicePipeline.cxx)

#------------------------------------------------------------------------------
find_package(ParaView QUIET COMPONENTS vtkPVCatalyst)
if(NOT ParaView_FOUND)
  message(FATAL_ERROR "Catalyst analysis components require Catalyst build (or "
                      "install directory. Please set ParaView_DIR to point to "
                      "directory containing `ParaViewConfig.cmake`.")
endif()
add_library(catalyst STATIC ${sources})
target_include_directories(catalyst
                            SYSTEM PUBLIC ${PARAVIEW_INCLUDE_DIRS}
                            INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(catalyst
                            PUBLIC ${VTK_DEFINITIONS}
                            INTERFACE ENABLE_CATALYST)
target_link_libraries(catalyst
                        PUBLIC ${VTK_LIBRARIES}
                        PUBLIC core)

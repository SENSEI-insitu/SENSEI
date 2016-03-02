set(sources
  vtkADIOSAnalysisAdaptor.cxx
  vtkADIOSDataAdaptor.cxx
  )

#------------------------------------------------------------------------------
find_package(ADIOS 1.9 REQUIRED)

add_library(adios STATIC ${sources})
target_include_directories(adios
  SYSTEM PRIVATE ${ADIOS_INCLUDE_DIRS}
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adios PRIVATE ${ADIOS_LIBRARIES} PUBLIC core PRIVATE mpi)
if(ADIOS_DEFINITIONS)
  target_compile_definitions(adios PRIVATE ${ADIOS_DEFINITIONS})
endif()

#------------------------------------------------------------------------------
add_executable(ADIOSAnalysisEndPoint ADIOSAnalysisEndPoint.cxx)
target_link_libraries(ADIOSAnalysisEndPoint PRIVATE adios opts mpi ${ADIOS_LIBRARIES} configurable)
target_include_directories(ADIOSAnalysisEndPoint SYSTEM PRIVATE ${ADIOS_INCLUDE_DIRS})
if(ADIOS_DEFINITIONS)
  target_compile_definitions(ADIOSAnalysisEndPoint PRIVATE ${ADIOS_DEFINITIONS})
endif()

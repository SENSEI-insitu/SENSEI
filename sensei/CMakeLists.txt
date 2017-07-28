if (ENABLE_SENSEI)
  message(STATUS "Enabled: sensei library")

  set(sensei_sources Error.cxx AnalysisAdaptor.cxx Autocorrelation.cxx
    ConfigurableAnalysis.cxx DataAdaptor.cxx PosthocIO.cxx Histogram.cxx
    VTKHistogram.cxx VTKDataAdaptor.cxx)

  set(sensei_libs mpi pugixml vtk thread ArrayIO timer diy grid)

  if(ENABLE_CINEMA)
    list(APPEND sensei_sources CinemaHelper.cxx VTKmContourCompositeAnalysis.cxx)
  endif()

  if(ENABLE_CATALYST)
    list(APPEND sensei_sources CatalystAnalysisAdaptor.cxx
      CatalystSlice.cxx CatalystUtilities.cxx)

    if(ENABLE_CINEMA)
      list(APPEND sensei_sources CatalystCinema.cxx)
    endif()
  endif()

  if(ENABLE_ADIOS)
    list(APPEND sensei_sources ADIOSAnalysisAdaptor.cxx
      ADIOSDataAdaptor.cxx)
    list(APPEND sensei_libs adios)
  endif()

  if(ENABLE_VTK_M)
    list(APPEND sensei_sources
      VTKmContourAnalysis.cxx
      VTKmContourAnalysis.h)
  endif()

  if(ENABLE_LIBSIM)
    list(APPEND sensei_sources LibsimAnalysisAdaptor.cxx
      LibsimImageProperties.cxx)
    list(APPEND sensei_libs libsim)
  endif()

  if(ENABLE_VTK_XMLP)
    list(APPEND sensei_sources VTKPosthocIO.cxx)
  endif()

  add_library(sensei ${sensei_sources})

  target_include_directories(sensei PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(sensei PUBLIC ${sensei_libs})

  if (ENABLE_VTK_M)
    target_compile_definitions(sensei PUBLIC ENABLE_VTK_M)
  endif()

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION include FILES_MATCHING PATTERN "*.h")


  install(TARGETS sensei EXPORT sensei INCLUDES DESTINATION include
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(EXPORT sensei DESTINATION lib/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

endif()

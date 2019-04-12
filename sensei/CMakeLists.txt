if (ENABLE_SENSEI)
  message(STATUS "Enabled: sensei library")

  # senseiCore
  # everything but the Python and configurable analysis adaptors.
  set(senseiCore_sources AnalysisAdaptor.cxx Autocorrelation.cxx
    BinaryStream.cxx BlockPartitioner.cxx ConfigurableInTransitDataAdaptor.cxx
    ConfigurablePartitioner.cxx DataAdaptor.cxx DataRequirements.cxx Error.cxx
    Histogram.cxx InTransitAdaptorFactory.cxx InTransitDataAdaptor.cxx
    MappedPartitioner.cxx MeshMetadata.cxx MeshMetadataMap.cxx
    PlanarPartitioner.cxx ProgrammableDataAdaptor.cxx VTKHistogram.cxx
    VTKDataAdaptor.cxx VTKUtils.cxx XMLUtils.cxx)

  set(senseiCore_libs pugixml thread timer sDIY sVTK sMPI)

  if (ENABLE_CONDUIT)
    list(APPEND senseiCore_sources ConduitDataAdaptor.cxx)
    list(APPEND senseiCore_libs sConduit)
  endif()

  if (ENABLE_CATALYST)
    list(APPEND senseiCore_sources CatalystAnalysisAdaptor.cxx
      CatalystParticle.cxx CatalystSlice.cxx CatalystUtilities.cxx)
  endif()

  if (ENABLE_VTKM)
    list(APPEND senseiCore_sources VTKmVolumeReductionAnalysis.cxx
      VTKmCDFAnalysis.cxx CDFReducer.cxx CinemaHelper.cxx)
    list(APPEND senseiCore_libs sVTKm)
  endif()

  if (ENABLE_ADIOS1)
    list(APPEND senseiCore_sources ADIOS1Schema.cxx ADIOS1AnalysisAdaptor.cxx
      ADIOS1DataAdaptor.cxx)
    list(APPEND senseiCore_libs sADIOS1)
  endif()

 if (ENABLE_HDF5)
       list(APPEND senseiCore_sources HDF5DataAdaptor.cxx HDF5AnalysisAdaptor.cxx
        HDF5Schema.cxx)
    list(APPEND senseiCore_libs sHDF5)
  endif()

  if (ENABLE_VTK_ACCELERATORS)
    list(APPEND senseiCore_sources VTKmContourAnalysis.cxx)
  endif()

  if (ENABLE_LIBSIM)
    list(APPEND senseiCore_sources LibsimAnalysisAdaptor.cxx
      LibsimImageProperties.cxx)
    list(APPEND senseiCore_libs sLibsim)
  endif()

  if (ENABLE_VTK_IO)
    list(APPEND senseiCore_sources VTKPosthocIO.cxx)
    if (ENABLE_VTK_MPI)
      list(APPEND senseiCore_sources VTKAmrWriter.cxx)
    endif()
  endif()

  add_library(senseiCore ${senseiCore_sources})

  target_include_directories(senseiCore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(senseiCore PUBLIC ${senseiCore_libs})

  install(TARGETS senseiCore EXPORT senseiCore
    INCLUDES DESTINATION include ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

  install(EXPORT senseiCore DESTINATION lib/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

  set(sensei_sources ConfigurableAnalysis.cxx)
  set(sensei_libs senseiCore)

  # PythonAnalysis
  if (ENABLE_PYTHON)
    depend_swig(PythonAnalysis.i PythonAnalysis.dep)

    wrap_swig(PythonAnalysis.i _PythonAnalysis.cxx
      PythonAnalysis.dep _PythonAnalysis)

    include_directories(SYSTEM ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}
      ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/python)

    add_library(_PythonAnalysis SHARED
      ${CMAKE_CURRENT_BINARY_DIR}/_PythonAnalysis.cxx)

    set_target_properties(_PythonAnalysis PROPERTIES PREFIX "")
    target_link_libraries(_PythonAnalysis PUBLIC senseiCore sPython)

    add_custom_command(TARGET _PythonAnalysis POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/PythonAnalysis.py
        ${CMAKE_CURRENT_BINARY_DIR}/../lib)

    install(TARGETS _PythonAnalysis EXPORT _PythonAnalysis
      LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

    install(EXPORT _PythonAnalysis DESTINATION lib/cmake
      EXPORT_LINK_INTERFACE_LIBRARIES)

    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/PythonAnalysis.py
      DESTINATION lib
    )

    set(python_analyses Histogram.py)
    foreach(python_analysis ${python_analyses})
      configure_file(${python_analysis}
        ${CMAKE_CURRENT_BINARY_DIR}/../lib/${python_analysis} COPYONLY)
    endforeach()

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
      DESTINATION lib FILES_MATCHING PATTERN "*.py")

    list(APPEND sensei_libs _PythonAnalysis)
  endif()

  # sensei
  # world facing library that users will link to
  # contains senseiCore classes, PythonAnalysis, and ConfigurableAnalysis
  add_library(sensei ${sensei_sources})

  target_include_directories(sensei PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(sensei PUBLIC ${sensei_libs})

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION include FILES_MATCHING PATTERN "*.h")

  install(TARGETS sensei EXPORT sensei
    INCLUDES DESTINATION include ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

  install(EXPORT sensei DESTINATION lib/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

  add_subdirectory(testing)
endif()

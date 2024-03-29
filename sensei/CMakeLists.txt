if (SENSEI_ENABLE_SENSEI)
  message(STATUS "SENSEI: Enabled: sensei library")

  # senseiCore
  # everything but the Python and configurable analysis adaptors.
  set(senseiCore_sources AnalysisAdaptor.cxx Autocorrelation.cxx
    BinaryStream.cxx BlockPartitioner.cxx ConfigurableInTransitDataAdaptor.cxx
    ConfigurablePartitioner.cxx DataAdaptor.cxx DataBinning.cxx
    DataRequirements.cxx Error.cxx Histogram.cxx HistogramInternals.cxx
    InTransitAdaptorFactory.cxx InTransitDataAdaptor.cxx
    IsoSurfacePartitioner.cxx MappedPartitioner.cxx MemoryProfiler.cxx
    MemoryUtils.cxx MeshMetadata.cxx MeshMetadataMap.cxx MPIManager.cxx
    ParticleDensity.cxx PlanarPartitioner.cxx PlanarSlicePartitioner.cxx
    Profiler.cxx ProgrammableDataAdaptor.cxx SVTKDataAdaptor.cxx SVTKUtils.cxx
    XMLUtils.cxx)

  set(senseiCore_libs pugixml thread sDIY sSVTK sMPI hamr)

  if (SENSEI_ENABLE_CONDUIT)
    list(APPEND senseiCore_sources ConduitDataAdaptor.cxx)
    list(APPEND senseiCore_libs sConduit)
  endif()

  if(SENSEI_ENABLE_ASCENT)
    list(APPEND senseiCore_sources AscentAnalysisAdaptor.cxx)
    list(APPEND senseiCore_libs sAscent)
  endif()

  if(SENSEI_ENABLE_CATALYST)
    list(APPEND senseiCore_sources CatalystAnalysisAdaptor.cxx
      CatalystParticle.cxx CatalystSlice.cxx CatalystUtilities.cxx)
  endif()

  if(SENSEI_ENABLE_CATALYST2)
    list(APPEND senseiCore_sources Catalyst2AnalysisAdaptor.cxx svtkDataObjectToConduit.cxx)
    list(APPEND senseiCore_libs sCatalyst2)
  endif()

  if (SENSEI_ENABLE_VTKM)  
    list(APPEND senseiCore_sources VTKmVolumeReductionAnalysis.cxx
      VTKmCDFAnalysis.cxx CDFReducer.cxx CinemaHelper.cxx)
    list(APPEND senseiCore_libs sVTKm)
  endif()

  if (SENSEI_ENABLE_ADIOS1)
    list(APPEND senseiCore_sources ADIOS1Schema.cxx ADIOS1AnalysisAdaptor.cxx
      ADIOS1DataAdaptor.cxx)
    list(APPEND senseiCore_libs sADIOS1)
  endif()

if (SENSEI_ENABLE_ADIOS2)
    list(APPEND senseiCore_sources ADIOS2Schema.cxx ADIOS2AnalysisAdaptor.cxx
      ADIOS2DataAdaptor.cxx)
    list(APPEND senseiCore_libs sADIOS2)
  endif()

 if (SENSEI_ENABLE_HDF5)
       list(APPEND senseiCore_sources HDF5DataAdaptor.cxx HDF5AnalysisAdaptor.cxx
        HDF5Schema.cxx)
    list(APPEND senseiCore_libs sHDF5)
  endif()

  if (SENSEI_ENABLE_OSPRAY)
    list(APPEND senseiCore_sources OSPRayAnalysisAdaptor.cxx)
    list(APPEND senseiCore_libs sOSPRay)
  endif()

  if (SENSEI_ENABLE_VTK_ACCELERATORS)
    list(APPEND senseiCore_sources VTKmContourAnalysis.cxx)
  endif()

  if (SENSEI_ENABLE_LIBSIM)
    list(APPEND senseiCore_sources LibsimAnalysisAdaptor.cxx
      LibsimImageProperties.cxx)
    list(APPEND senseiCore_libs sLibsim)
  endif()

  if (SENSEI_ENABLE_VTK_IO)
    list(APPEND senseiCore_sources VTKPosthocIO.cxx)
    if (SENSEI_ENABLE_VTK_MPI)
      list(APPEND senseiCore_sources VTKAmrWriter.cxx)
    endif()
    if (SENSEI_ENABLE_VTK_FILTERS)
      list(APPEND senseiCore_sources SliceExtract.cxx)
    endif()
  endif()

  if (SENSEI_ENABLE_VTK_FILTERS)
    list(APPEND senseiCore_sources Calculator.cxx)
  endif()

  if (SENSEI_ENABLE_VTK_CORE)
    list(APPEND senseiCore_libs sVTK)
  endif()

  set(senseiCore_cuda_sources)
  if (SENSEI_ENABLE_CUDA)
    list(APPEND senseiCore_cuda_sources CUDAUtils.cxx MemoryUtils.cxx
      DataBinning.cxx HistogramInternals.cxx ParticleDensity.cxx
      SVTKUtils.cxx)
  endif()

  if (SENSEI_ENABLE_OPENMP)
    list(APPEND senseiCore_libs OpenMP::OpenMP_CXX)
  endif()

  if (SENSEI_SEPARATE_IMPL)
    if (SENSEI_ENABLE_CUDA)
      list(APPEND senseiCore_cuda_sources svtkHAMRDataArray.cxx)
    else()
      list(APPEND senseiCore_sources svtkHAMRDataArray.cxx)
    endif()
  endif()

  add_target(LIB
    NAME senseiCore
    CXX_SOURCES ${senseiCore_sources}
    CU_SOURCES ${senseiCore_cuda_sources}
    CU_ARCH "${SENSEI_CUDA_ARCHITECTURES}"
    LINK_LIBS ${senseiCore_libs}
    LINK_OPTS ${SENSEI_OPENMP_LINK_FLAGS}
    )

  if (SENSEI_ENABLE_CATALYST)
    target_compile_definitions(senseiCore PRIVATE PARAVIEW_VERSION_MAJOR=${ParaView_VERSION_MAJOR} PARAVIEW_VERSION_MINOR=${ParaView_VERSION_MINOR})
  endif()

  if (SENSEI_ENABLE_VTK_IO)
    target_compile_definitions(senseiCore PRIVATE VTK_MAJOR_VERSION=${VTK_MAJOR_VERSION} VTK_MINOR_VERSION=${VTK_MINOR_VERSION} VTK_BUILD_VERSION=${VTK_BUILD_VERSION})
  endif()

  target_include_directories(senseiCore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(senseiCore PUBLIC ${senseiCore_libs})

  install(TARGETS senseiCore EXPORT senseiCore
    INCLUDES DESTINATION include ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(EXPORT senseiCore DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)

  set(sensei_sources ConfigurableAnalysis.cxx)
  set(sensei_libs senseiCore)

  # PythonAnalysis
  if (SENSEI_ENABLE_PYTHON)
    set_property(SOURCE PythonAnalysis.i PROPERTY CPLUSPLUS ON)
    set_property(SOURCE PythonAnalysis.i PROPERTY SWIG_MODULE_NAME PythonAnalysis)
    set_property(SOURCE PythonAnalysis.i PROPERTY DEPENDS DataAdaptor.i)

    swig_add_library(PythonAnalysis TYPE SHARED
      LANGUAGE python SOURCES PythonAnalysis.i
      OUTPUT_DIR "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}"
      OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    set_target_properties(PythonAnalysis PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}"
      SUFFIX .so)

    target_link_libraries(PythonAnalysis senseiCore pysvtk sPython)

    target_include_directories(PythonAnalysis PUBLIC
      "${CMAKE_SOURCE_DIR}/python" "${CMAKE_SOURCE_DIR}/utils/pysvtk"
      "${NUMPY_INCLUDE_DIR}")

    set_property(TARGET PythonAnalysis
      PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

    if (SENSEI_ENABLE_OPENMP AND SENSEI_ENABLE_CUDA AND NOT SENSEI_NVHPC_CUDA)
      target_link_options(PythonAnalysis PUBLIC ${SENSEI_OPENMP_LINK_FLAGS})
    endif ()

    # install the Python module
    install(TARGETS PythonAnalysis DESTINATION ${SENSEI_PYTHON_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/PythonAnalysis.py
      DESTINATION ${SENSEI_PYTHON_DIR})

    # include the directory containing _PythonAnalysis in the rpath this
    # enables one not to have to set (DY)LD_LIBRARY_PATH prior to importing
    # sensei.
    if (APPLE)
      set_target_properties(PythonAnalysis PROPERTIES INSTALL_RPATH "@loader_path/./:@loader_path/../svtk")
    elseif(UNIX)
      set_target_properties(PythonAnalysis PROPERTIES INSTALL_RPATH "\$ORIGIN/./:\$ORIGIN/../svtk")
    endif()

    add_library(sPythonAnalysis SHARED IMPORTED)
    set_property(TARGET sPythonAnalysis
      PROPERTY IMPORTED_LOCATION
      "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/_PythonAnalysis.so")
    if (APPLE)
      set_property(TARGET sPythonAnalysis
      PROPERTY IMPORTED_SONAME "@rpath/_PythonAnalysis.so")
    endif()
    add_dependencies(sPythonAnalysis PythonAnalysis)

    # install Python analyses
    set(python_analyses Histogram.py)
    foreach(python_analysis ${python_analyses})
      configure_file(${python_analysis}
        "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/${python_analysis}"
        COPYONLY)
      install(FILES ${python_analysis} DESTINATION ${SENSEI_PYTHON_DIR})
    endforeach()

    # add to the sensei library so link info gets propagated
    list(APPEND sensei_libs sPythonAnalysis)
  endif()

  # sensei
  # world facing library that users will link to
  # contains senseiCore classes, PythonAnalysis, and ConfigurableAnalysis
  add_target(LIB
    NAME sensei
    CXX_SOURCES ${sensei_sources}
    LINK_LIBS ${sensei_libs}
    LINK_OPTS ${SENSEI_OPENMP_LINK_FLAGS}
    )
  #add_library(sensei ${sensei_sources})
  #target_link_libraries(sensei PUBLIC ${sensei_libs})

  target_include_directories(sensei PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION include FILES_MATCHING PATTERN "*.h")

  install(TARGETS sensei EXPORT sensei
    INCLUDES DESTINATION include ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(EXPORT sensei DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)

  add_subdirectory(testing)
endif()

if (BUILD_TESTING)

  ##############################################################################
  senseiAddTest(testHistogramSerial
    SOURCES testHistogram.cpp LIBS sensei EXEC_NAME testHistogram
    COMMAND $<TARGET_NAME:testHistogram>
    LABELS HISTO)

  senseiAddTest(testHistogramParallel
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_NAME:testHistogram>
    PROPERTIES
      LABELS HISTO)

  ##############################################################################
  senseiAddTest(testHDF5Write
    SOURCES testHDF5.cpp LIBS sensei EXEC_NAME testHDF5
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_NAME:testHDF5> w 4 n h5test
    FEATURES HDF5)

  senseiAddTest(testHDF5Read
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_NAME:testHDF5> r h5test.n${TEST_NP}
    FEATURES HDF5
    PROPERTIES
      DEPENDS testHDF5Write)

  ##############################################################################
  senseiAddTest(testHDF5WriteStreaming
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_NAME:testHDF5> w 4 s h5stream
    FEATURES HDF5
    PROPERTIES
      LABELS STREAMING)

  senseiAddTest(testHDF5ReadStreaming
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_NAME:testHDF5> r h5stream.n${TEST_NP} s
    FEATURES HDF5
    PROPERTIES
      DEPENDS testHDF5WriteStreaming
      LABELS STREAMING)

  ##############################################################################
  senseiAddTest(testProgrammableDataAdaptor
    PARALLEL 1
    COMMAND $<TARGET_NAME:testProgrammableDataAdaptor>
    SOURCES testProgrammableDataAdaptor.cpp
    LIBS sensei)

  senseiAddTest(testProgrammableDataAdaptorPy
    PARALLEL 1
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/testProgrammableDataAdaptor.py
    FEATURES PYTHON)

  ##############################################################################
  senseiAddTest(testPythonAnalysis
    SOURCES testPythonAnalysis.cpp LIBS sensei EXEC_NAME testPythonAnalysis
    COMMAND
      $<TARGET_NAME:testPythonAnalysis> ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml
    FEATURES PYTHON VTK_IO)

  senseiAddTest(testPythonAnalysisParallel
    PARALLEL ${TEST_NP}
    COMMAND
      $<TARGET_NAME:testPythonAnalysis> ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml
    FEATURES PYTHON VTK_IO)

  ##############################################################################
  senseiAddTest(testPartitionerPy
    COMMAND
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioner.py 13
    FEATURES PYTHON)

  ##############################################################################
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/catalyst_render_partition.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/catalyst_render_partition.xml  @ONLY)

  senseiAddTest(testPartitionersADIOS1BP
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} adios1_bp 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1 CATALYST)

  # in adios 1.13.1 flexpath has a limitation of 1 sender block per
  # MPI rank. there's also a deadlock if some blocks are not sent.
  # hence this test current has to run with M=9 N=2
  senseiAddTest(testPartitionersADIOS1FLEXPATH
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 2
      ${CMAKE_CURRENT_SOURCE_DIR} adios1_flexpath 3 3 3 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1 CATALYST)

  senseiAddTest(testADIOS1Flexpath
    PARALLEL_SHELL ${TEST_NP}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1Flexpath.bp FLEXPATH FLEXPATH 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1)

  senseiAddTest(testADIOS1MPIBP
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1MPIBP.bp MPI BP 3
    FEATURES
      PYTHON ADIOS1)

  ##############################################################################
  senseiAddTest(testADIOS2BP4
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS2.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS2BP4_%05d%.bp BP4 BP4 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2)

  senseiAddTest(testADIOS2SST
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS2.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS2SST.bp SST SST 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2)

  senseiAddTest(testPartitionersADIOS2BP4
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} adios2_bp4 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2 CATALYST)

  senseiAddTest(testPartitionersADIOS2SST
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} adios2_sst 3 3 3 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2 CATALYST)

  ##############################################################################
  senseiAddTest(testPartitionersHDF5A
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} h5 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON HDF5 CATALYST)

  senseiAddTest(testPartitionersHDF5B
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} h6 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON HDF5 CATALYST)

  ##############################################################################
  senseiAddTest(testADIOS1FLEXPATHHistogram
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioners.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 2 2 2
      ${CMAKE_CURRENT_SOURCE_DIR} write_adios1_flexpath.xml
      histogram.xml read_adios1_flexpath_default.xml 10 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1)

  ##############################################################################
  senseiAddTest(testADIOS2SSTHistogram
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioners.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 4 4 2
      ${CMAKE_CURRENT_SOURCE_DIR} write_adios2_sst.xml
      histogram.xml read_adios2_sst_block.xml 10 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2)

  ##############################################################################
  senseiAddTest(testMeshMetadata
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testMeshMetadata.py
    FEATURES PYTHON)

endif()

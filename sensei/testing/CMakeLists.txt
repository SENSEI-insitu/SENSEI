if (BUILD_TESTING)
  senseiAddTest(testHistogramSerial
    COMMAND testHistogram EXEC_NAME testHistogram
    SOURCES testHistogram.cpp LIBS sensei)

  senseiAddTest(testHistogramParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${MPIEXEC_POSTFLAGS} testHistogram)

  senseiAddTest(testADIOS1Flexpath
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1Flexpath.bp FLEXPATH FLEXPATH 3
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1})

  senseiAddTest(testADIOS1MPIBP
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1MPIBP.bp MPI BP 3
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1})

  senseiAddTest(testHDF5Write
    SOURCES testHDF5.cpp LIBS sensei EXEC_NAME testHDF5
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
    testHDF5 w 4 n h5test
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5Read
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
    testHDF5 r h5test.n${TEST_NP}
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5WriteStreaming
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
    testHDF5 w 4 s h5stream
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5ReadStreaming
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
    testHDF5 r  h5stream.n${TEST_NP} s
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testProgrammableDataAdaptor
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
      ${MPIEXEC_POSTFLAGS} testProgrammableDataAdaptor
    SOURCES testProgrammableDataAdaptor.cpp
    LIBS sensei)

  senseiAddTest(testProgrammableDataAdaptorPy
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
      ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/testProgrammableDataAdaptor.py
    FEATURES ${ENABLE_PYTHON})

  senseiAddTest(testPythonAnalysis
    COMMAND testPythonAnalysis
    ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml
    EXEC_NAME testPythonAnalysis SOURCES testPythonAnalysis.cpp
    LIBS sensei
    FEATURES ${ENABLE_PYTHON})

  senseiAddTest(testPythonAnalysisParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${MPIEXEC_POSTFLAGS} testPythonAnalysis
      ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml
    FEATURES ${ENABLE_PYTHON})

  senseiAddTest(testPartitionerPy
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioner.py 13
    FEATURES ${ENABLE_PYTHON})

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/catalyst_render_partition.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/catalyst_render_partition.xml  @ONLY)

  senseiAddTest(testPartitionersADIOS1BP
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP} ${TEST_NP_HALF}
      ${CMAKE_CURRENT_SOURCE_DIR} adios1_bp 3 3 3 0
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1} ${ENABLE_CATALYST})

  # in adios 1.13.1 flexpath has a limitation of 1 sender block per
  # MPI rank. there's also a deadlock if some blocks are not sent.
  # hence this test current has to run with M=9 N=2
  senseiAddTest(testPartitionersADIOS1FLEXPATH
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 2
      ${CMAKE_CURRENT_SOURCE_DIR} adios1_flexpath 3 3 3 1
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1} ${ENABLE_CATALYST})

  senseiAddTest(testPartitionersHDF5A
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP} ${TEST_NP_HALF}
      ${CMAKE_CURRENT_SOURCE_DIR} h5 3 3 3 0
    FEATURES ${ENABLE_PYTHON} ${ENABLE_HDF5} ${ENABLE_CATALYST})

  senseiAddTest(testPartitionersHDF5B
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP} ${TEST_NP_HALF}
      ${CMAKE_CURRENT_SOURCE_DIR} h6 3 3 3 0
    FEATURES ${ENABLE_PYTHON} ${ENABLE_HDF5} ${ENABLE_CATALYST})

  senseiAddTest(testADIOS1FLEXPATHHistogram
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioners.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 2 2 2
      ${CMAKE_CURRENT_SOURCE_DIR} write_adios1_flexpath.xml
      histogram.xml read_adios1_flexpath_default.xml 10 1
    FEATURES  ${ENABLE_PYTHON} ${ENABLE_ADIOS1})

endif()

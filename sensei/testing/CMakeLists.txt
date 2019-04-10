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

  senseiAddTest(setupHDF5
    COMMAND testHDF5 EXEC_NAME testHDF5
    SOURCES testHDF5.cpp LIBS sensei
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5Write
    COMMAND testHDF5 w 4 n h5test
    SOURCES testHDF5.cpp LIBS sensei
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5Read
    COMMAND testHDF5 r h5test.n1
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5WriteStreaming
    COMMAND testHDF5 w 4 s h5stream
    FEATURES ${ENABLE_HDF5})

  senseiAddTest(testHDF5ReadStreaming
    COMMAND testHDF5 r h5stream.n1 s
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
endif()

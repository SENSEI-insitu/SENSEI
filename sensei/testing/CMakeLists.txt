if (BUILD_TESTING)

  senseiAddTest(testHistogramSerial
    COMMAND testHistogram EXEC_NAME testHistogram
    SOURCES testHistogram.cpp LIBS sensei)

  senseiAddTest(testHistogramParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_POSTFLAGS} testHistogram)

  senseiAddTest(testADIOS1Flexpath
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1Flexpath.bp FLEXPATH FLEXPATH 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1})

  senseiAddTest(testADIOS1MPIBP
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1MPIBP.bp MPI BP 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1})

  senseiAddTest(testProgrammableDataAdaptor
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
      ${MPIEXEC_POSTFLAGS} testProgrammableDataAdaptor
    SOURCES testProgrammableDataAdaptor.cpp
    LIBS sensei)

  senseiAddTest(testProgrammableDataAdaptorPy
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
      ${MPIEXEC_POSTFLAGS} python
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
      ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_POSTFLAGS} testPythonAnalysis
      ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml
    FEATURES ${ENABLE_PYTHON})
endif()

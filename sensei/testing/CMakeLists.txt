if (BUILD_TESTING)

  senseiAddTest(testHistogramSerial
    COMMAND testHistogram EXEC_NAME testHistogram
    SOURCES testHistogram.cpp LIBS sensei)

  senseiAddTest(testHistogramParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_POSTFLAGS} testHistogram)

  senseiAddTest(testADIOSFlexpath
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOSFlexpath.bp FLEXPATH FLEXPATH 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})

  senseiAddTest(testADIOSMPIBP
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOSMPIBP.bp MPI BP 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})

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
    LIBS sensei)

  senseiAddTest(testPythonAnalysisParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_POSTFLAGS} testPythonAnalysis
      ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml)
endif()

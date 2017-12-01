if (BUILD_TESTING)

  senseiAddTest(testHistogramSerial
    COMMAND testHistogram EXEC_NAME testHistogram
    SOURCES testHistogram.cpp LIBS sensei)

  senseiAddTest(testHistogramParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} testHistogram)

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

endif()

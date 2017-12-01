if (BUILD_TESTING)

  sensei_add_test(testHistogramSerial
    COMMAND testHistogram EXEC_NAME testHistogram
    SOURCES testHistogram.cpp LIBS sensei)

  sensei_add_test(testHistogramParallel
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} testHistogram)

  sensei_add_test(testADIOSFlexpath
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOSFlexpath.bp FLEXPATH FLEXPATH 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})

  sensei_add_test(testADIOSMPIBP
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOSMPIBP.bp MPI BP 2
    FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})

endif()

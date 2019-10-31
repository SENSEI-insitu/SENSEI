if (BUILD_TESTING)

  senseiAddTest(testOscillatorHistogram
    COMMAND oscillator -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_histogram.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorHistogramPar
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
     oscillator -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_histogram.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorAutocorrelation
    COMMAND oscillator -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_autocorrelation.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorAutocorrelationPar
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
     oscillator -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_autocorrelation.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  if (ENABLE_VTK_IO)
    senseiAddTest(testOscillatorVTKWriter
      COMMAND oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_vtkwriter.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

    senseiAddTest(testOscillatorVTKWriterPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
       oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_vtkwriter.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)
  endif()

  if (ENABLE_CATALYST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml @ONLY)

    senseiAddTest(testOscillatorCatalyst
      COMMAND oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

    senseiAddTest(testOscillatorCatalystPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
       oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)
  endif()

  if (ENABLE_LIBSIM)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml @ONLY)

    senseiAddTest(testOscillatorLibsim
      COMMAND oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

    senseiAddTest(testOscillatorLibsimPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
       oscillator -t 1 -b ${TEST_NP} -g 1
        -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml
        ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)
  endif()

  # TODO -- this test breaks dashboard builds
  #if (ENABLE_CATALYST)
  #  add_test(NAME testCatalystSlice
  #    COMMAND ${CMAKE_COMMAND}
  #     -DCATALYST_TEST_DRIVER:FILEPATH=$<TARGET_FILE:oscillator>
  #    -DIMAGE_TESTER:FILEPATH=$<TARGET_FILE:CompareImages>
  #    -DCATALYST_TEST_DIR:PATH=${CMAKE_BINARY_DIR}/Testing
  #    -DCATALYST_TEST_DATA=${CMAKE_CURRENT_SOURCE_DIR}
  #    -DMPIEXEC:FILEPATH=${MPIEXEC}
  #    -DMPIEXEC_NUMPROC_FLAG:STRING=${MPIEXEC_NUMPROC_FLAG}
  #    -DMPIEXEC_NUMPROCS=2
  #    -DMPIEXEC_PREFLAGS:STRING=${MPIEXEC_PREFLAGS}
  #    -DVTK_MPI_POSTFLAGS:STRING=${VTK_MPI_POSTFLAGS}
  #    -P ${CMAKE_CURRENT_SOURCE_DIR}/CatalystSlice.cmake)
  #endif()

endif()

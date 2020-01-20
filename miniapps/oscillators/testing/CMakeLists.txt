if (BUILD_TESTING)

  senseiAddTest(testOscillatorHistogram
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_histogram.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorHistogramPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_histogram.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  if (ENABLE_PYTHON)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_python_histogram.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_python_histogram.xml @ONLY)
  endif()

  senseiAddTest(testOscillatorPythonHistogram
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_python_histogram.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
      FEATURES PYTHON)

  senseiAddTest(testOscillatorAutocorrelation
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_autocorrelation.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorAutocorrelationPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_autocorrelation.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc)

  senseiAddTest(testOscillatorVTKWriter
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_vtkwriter.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES VTK_IO)

  senseiAddTest(testOscillatorVTKWriterPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_vtkwriter.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES VTK_IO)

  if (ENABLE_CATALYST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml @ONLY)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst_bidirectional.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst_bidirectional.xml @ONLY)
  endif()

  senseiAddTest(testOscillatorCatalyst
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES CATALYST)

  senseiAddTest(testOscillatorCatalystPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES CATALYST)

  senseiAddTest(testOscillatorCatalystBiDirectional
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst_bidirectional.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES CATALYST)

  senseiAddTest(testOscillatorCatalystBiDirectionalPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_catalyst_bidirectional.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES CATALYST)

  if (ENABLE_LIBSIM)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml @ONLY)
  endif()

  senseiAddTest(testOscillatorLibsim
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES LIBSIM)

  senseiAddTest(testOscillatorLibsimPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 1 -b ${TEST_NP} -g 1
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_libsim.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/simple.osc
    FEATURES LIBSIM)

  if (ENABLE_ASCENT)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oscillator_ascent.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_ascent.xml @ONLY)
  endif()

  senseiAddTest(testOscillatorAscent
    COMMAND $<TARGET_FILE:oscillator> -t 2 -b ${TEST_NP} -g 0
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_ascent.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/random_2d_64.osc
    FEATURES ASCENT)

  senseiAddTest(testOscillatorAscentPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:oscillator> -t 2 -b ${TEST_NP} -g 0
      -f ${CMAKE_CURRENT_SOURCE_DIR}/oscillator_ascent.xml
      ${CMAKE_CURRENT_SOURCE_DIR}/random_2d_64.osc
    FEATURES ASCENT)

  # TODO -- this test breaks dashboard builds
  #if (ENABLE_CATALYST)
  #  add_test(NAME testCatalystSlice
  #    COMMAND ${CMAKE_COMMAND}
  #     -DCATALYST_TEST_DRIVER:FILEPATH=$<TARGET_FILE:$<TARGET_FILE:oscillator>>
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

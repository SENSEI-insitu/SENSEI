if (BUILD_TESTING)

  senseiAddTest(testMandelbrotHistogram
    COMMAND mandelbrot -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_histogram.xml)

  senseiAddTest(testMandelbrotHistogramPar
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
     mandelbrot -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_histogram.xml)

  if (ENABLE_VTK_IO)
    senseiAddTest(testMandelbrotVTKWriter
      COMMAND mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_vtkwriter.xml)

    senseiAddTest(testMandelbrotVTKWriterPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
        mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_vtkwriter.xml)
  endif()

  if (ENABLE_CATALYST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml @ONLY)

    senseiAddTest(testMandelbrotCatalyst
      COMMAND
        mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml)

    senseiAddTest(testMandelbrotCatalystPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
        mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml)
  endif()

  if (ENABLE_LIBSIM)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml @ONLY)

    senseiAddTest(testMandelbrotLibsim
      COMMAND mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml)

    senseiAddTest(testMandelbrotLibsimPar
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
        mandelbrot -i 2 -l 2
        -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml)
  endif()

endif()

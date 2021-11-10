if (BUILD_TESTING)

  senseiAddTest(testMandelbrotHistogram
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_histogram.xml)

  senseiAddTest(testMandelbrotHistogramPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_histogram.xml)

  senseiAddTest(testMandelbrotVTKWriter
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_vtkwriter.xml
    FEATURES VTK_IO VTK_MPI)

  senseiAddTest(testMandelbrotVTKWriterPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_vtkwriter.xml
    FEATURES VTK_IO VTK_MPI)

  if (ENABLE_CATALYST)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml @ONLY)
  endif()

  senseiAddTest(testMandelbrotCatalyst
    COMMAND
      $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml
    FEATURES CATALYST)

  senseiAddTest(testMandelbrotCatalystPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_catalyst.xml
      FEATURES CATALYST)

  if (ENABLE_LIBSIM)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml.in
      ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml @ONLY)
  endif()

  senseiAddTest(testMandelbrotLibsim
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml
    FEATURES LIBSIM)

  senseiAddTest(testMandelbrotLibsimPar
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:mandelbrot> -i 2 -l 2
      -f ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_libsim.xml
    FEATURES LIBSIM)

endif()

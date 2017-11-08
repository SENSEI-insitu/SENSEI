if (BUILD_TESTING)
  sensei_add_test(test_adios_flexpath
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_adios.sh 2
      ${CMAKE_CURRENT_SOURCE_DIR} test_adios_flexpath.bp
      FLEXPATH FLEXPATH 2
      FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})
  sensei_add_test(test_adios_mpi_bp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_adios.sh 2
      ${CMAKE_CURRENT_SOURCE_DIR} test_adios_mpi_bp.bp
      MPI BP 2
      FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS})
endif()

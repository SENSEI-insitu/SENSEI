if (PYTHON_EXECUTABLE AND NOT SVTK_BUILD_SHARED_LIBS)
  # For now, this test is only added for static builds.
  # In theory, we can make it work in shared builds too, it just needs more work to figure out
  # what variables need to be set on each platform correctly.
  add_test(NAME svtkPythonInterpreterPythonTests-TestImportVanillaInterpreter
    COMMAND "${PYTHON_EXECUTABLE}" -m svtk)
  set_tests_properties("svtkPythonInterpreterPythonTests-TestImportVanillaInterpreter"
    PROPERTIES
      LABELS "${_svtk_build_test_labels}"
      FAIL_REGULAR_EXPRESSION "${_svtk_fail_regex}"
      # This must match SVTK_SKIP_RETURN_CODE in svtkTestingObjectFactory.h
      SKIP_RETURN_CODE 125
      ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}/_svtk.zip:${CMAKE_BINARY_DIR}/${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}/"
    )
endif()

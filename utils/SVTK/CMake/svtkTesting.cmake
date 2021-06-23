include(svtkExternalData)

# Test input data staging directory.
file(RELATIVE_PATH svtk_reldir "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set(SVTK_TEST_DATA_DIR "${ExternalData_BINARY_ROOT}/${svtk_reldir}/Testing")

# Test input data directory.
set(SVTK_TEST_INPUT_DIR "${SVTK_SOURCE_DIR}/Testing/Data")

# Test output directory.
set(SVTK_TEST_OUTPUT_DIR "${SVTK_BINARY_DIR}/Testing/Temporary")

if(NOT EXISTS "${SVTK_SOURCE_DIR}/.ExternalData/README.rst")
  # This file is always present in version-controlled source trees
  # so we must have been extracted from a source tarball with no
  # data objects needed for testing.  Turn off tests by default
  # since enabling them requires network access or manual data
  # store configuration.
  set(SVTK_BUILD_TESTING OFF)
endif()
include(CTest)
set_property(CACHE BUILD_TESTING
  PROPERTY
    TYPE INTERNAL)

# Provide an option for tests requiring "large" input data
option(SVTK_USE_LARGE_DATA "Enable tests requiring \"large\" data" OFF)

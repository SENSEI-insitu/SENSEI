include("${CMAKE_CURRENT_LIST_DIR}/github_actions.cmake")

# Read the files from the build directory.
ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")

# Pick up from where the configure left off.
ctest_start(APPEND)

include(ProcessorCount)
ProcessorCount(nproc)

# Default to a reasonable test timeout.
set(CTEST_TEST_TIMEOUT 100)

set(test_exclude_tests)
list(APPEND test_exclude_tests
  # Disabled for CI because it uses too many proccess
  "^testADIOS2SSTHistogram$"
  # HDF5 Adaptors need an update for Unstructured Grid
  "^testHDF5Read*"
)
string(REPLACE ";" "|" test_exclude_tests "${test_exclude_tests}")
if (test_exclude_tests)
  set(test_exclude_tests "(${test_exclude_tests})")
endif ()

set(test_exclude_labels)
string(REPLACE ";" "|" test_exclude_labels "${test_exclude_labels}")
if (test_exclude_labels)
  set(test_exclude_tests "(${test_exclude_labels})")
endif ()

if (APPLE)
set(ENV{DYLD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib:$ENV{DYLD_LIBRARY_PATH}")
set(ENV{DYLD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib64:$ENV{DYLD_LIBRARY_PATH}")
elseif (UNIX)
set(ENV{LD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib:$ENV{LD_LIBRARY_PATH}")
set(ENV{LD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib64:$ENV{LD_LIBRARY_PATH}")
endif ()

ctest_test(APPEND
  TEST_LOAD "${nproc}"
  RETURN_VALUE test_result
  EXCLUDE "${test_exclude_tests}"
  EXCLUDE_LABEL "${test_exclude_labels}")
if (DO_SUBMIT)
  ctest_submit(PARTS Test)
endif ()

if (test_result)
  message(FATAL_ERROR
    "Failed to test")
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/github_actions.cmake")

# Read the files from the build directory.
ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")

# Pick up from where the configure left off.
ctest_start(APPEND)

include(ProcessorCount)
ProcessorCount(nproc)
if (NOT "$ENV{CTEST_MAX_PARALLELISM}" STREQUAL "")
  if (nproc GREATER "$ENV{CTEST_MAX_PARALLELISM}")
    set(nproc "$ENV{CTEST_MAX_PARALLELISM}")
  endif ()
endif ()

# Default to a reasonable test timeout.
set(CTEST_TEST_TIMEOUT 100)

set(test_exclude_labels)
if (${nproc} LESS 4)
  list(APPEND test_exclude_labels "PARALLEL")
endif ()

if (test_exclude_labels)
  list(PREPEND test_exclude_labels EXCLUDE_LABEL)
endif ()

if (APPLE)
set(ENV{DYLD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib:$ENV{DYLD_LIBRARY_PATH}")
elseif (UNIX)
set(ENV{LD_LIBRARY_PATH} "${CTEST_BINARY_DIRECTORY}/lib:$ENV{LD_LIBRARY_PATH}")
endif ()
set(ENV{PYTHONPATH} "${CTEST_BINARY_DIRECTORY}/lib:$ENV{PYTHONPATH}")

ctest_test(APPEND
  PARALLEL_LEVEL "${nproc}"
  TEST_LOAD "${nproc}"
  RETURN_VALUE test_result
  ${test_exclude_labels}
  REPEAT UNTIL_FAIL:3)
#ctest_submit(PARTS Test)

if (test_result)
  message(FATAL_ERROR
    "Failed to test")
endif ()

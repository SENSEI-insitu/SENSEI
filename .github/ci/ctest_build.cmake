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

if (CTEST_CMAKE_GENERATOR STREQUAL "Unix Makefiles")
  set(CTEST_BUILD_FLAGS "-j${nproc} -l${nproc}")
elseif (CTEST_CMAKE_GENERATOR MATCHES "Ninja")
  set(CTEST_BUILD_FLAGS "-l${nproc}")
endif ()

set(targets_to_build "all")

set(num_warnings 0)
foreach (target IN LISTS targets_to_build)
  set(build_args)
  if (NOT target STREQUAL "all")
    list(APPEND build_args TARGET ${target})
  endif ()

  if (CTEST_CMAKE_GENERATOR MATCHES "Make")
    # Drop the `-i` flag without remorse.
    set(CTEST_BUILD_COMMAND "make ${CTEST_BUILD_FLAGS}")

    if (NOT target STREQUAL "all")
      string(APPEND CTEST_BUILD_COMMAND " ${target}")
    endif ()
  endif ()

  ctest_build(
    NUMBER_WARNINGS num_warnings_target
    RETURN_VALUE    build_result
    ${build_args})

  math(EXPR num_warnings "${num_warnings} + ${num_warnings_target}")

  #ctest_submit(PARTS Build)
endforeach ()

if (build_result)
  message(FATAL_ERROR
    "Failed to build")
endif ()

if ("$ENV{CTEST_NO_WARNINGS_ALLOWED}" AND num_warnings GREATER 0)
  message(FATAL_ERROR
    "Found ${num_warnings} warnings (treating as fatal).")
endif ()

if (NOT DEFINED "ENV{GITHUB_ACTIONS}")
  message(STATUS
    "This script is being run outside of GitHub-Actions.")
else ()
  message(STATUS
    "This script is being run inside of GitHub-Actions")
endif ()

# Set up the source and build paths.
if (NOT DEFINED "ENV{GITHUB_WORKSPACE}")
  get_filename_component(project_dir "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
  set(CTEST_SOURCE_DIRECTORY "${project_dir}")
else()
  set(CTEST_SOURCE_DIRECTORY "$ENV{GITHUB_WORKSPACE}")
  set(CTEST_SITE "github-actions")
endif()

set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/../build")

if ("$ENV{CMAKE_CONFIGURATION}" STREQUAL "")
  message(FATAL_ERROR
    "The CMAKE_CONFIGURATION environment variable is required to know what "
    "cache initialization file to use.")
endif ()

# Set the build metadata.

set(build_name_prefix)
if (DEFINED "ENV{GITHUB_EVENT_NAME}")
  if ("$ENV{GITHUB_EVENT_NAME}" MATCHES "pull_request")
    set(build_name_prefix "-pr")
  elseif ("$ENV{GITHUB_EVENT_NAME}" MATCHES "push")
    set(build_name_prefix "-branch")
  else ()
    set(build_name_prefix "-$ENV{GITHUB_EVENT_NAME}")
  endif ()
  if (DEFINED "ENV{GITHUB_REF}")
    get_filename_component(ref "$ENV{GITHUB_REF}" NAME)
    string(APPEND build_name_prefix "-${ref}")
  endif ()
endif ()

set(CTEST_BUILD_NAME "SENSEI${build_name_prefix}[$ENV{CMAKE_CONFIGURATION}]")

# Default to Debug builds.
if (NOT "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CTEST_BUILD_CONFIGURATION "$ENV{CMAKE_BUILD_TYPE}")
endif ()
if (NOT CTEST_BUILD_CONFIGURATION)
  set(CTEST_BUILD_CONFIGURATION "Debug")
endif ()

# Default to using Ninja.
if (NOT "$ENV{CMAKE_GENERATOR}" STREQUAL "")
  set(CTEST_CMAKE_GENERATOR "$ENV{CMAKE_GENERATOR}")
endif ()
if (NOT CTEST_CMAKE_GENERATOR)
  set(CTEST_CMAKE_GENERATOR "Ninja")
endif ()

if (NOT DEFINED "ENV{CTEST_MAX_PARALLELISM}")
  set(ENV{CTEST_MAX_PARALLELISM} 4)
endif ()
# Determine the track to submit to.
set(ctest_model "Continuous")
set(ctest_track "Experimental")
set(DO_SUBMIT True)

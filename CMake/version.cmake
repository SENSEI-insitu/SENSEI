set(SENSEI_VERSION "<DEFAULT>" CACHE STRING "SENSEI version")

# Default detection of version
if (SENSEI_VERSION STREQUAL "<DEFAULT>")
  # The default version is the current major release
  # This is required for computing version in released
  # source tarballs.
  set(tmp "v4.0.0")

  # If inside a git repo, attempt to inspect the tag instead
  # of using the hardcoded value
  if (EXISTS ${CMAKE_SOURCE_DIR}/.git)
    find_package(Git QUIET)
    if(GIT_FOUND)
      execute_process(COMMAND ${GIT_EXECUTABLE}
          --git-dir=${CMAKE_SOURCE_DIR}/.git describe --tags
          OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif ()

  # Override the CACHE variable for configure with the computed
  # version
  set(SENSEI_VERSION ${tmp})
endif()

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\1" SENSEI_VERSION_MAJOR ${SENSEI_VERSION})

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\2" SENSEI_VERSION_MINOR ${SENSEI_VERSION})

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\3" SENSEI_VERSION_PATCH ${SENSEI_VERSION})

message(STATUS "SENSEI_VERSION_MAJOR=${SENSEI_VERSION_MAJOR}")
message(STATUS "SENSEI_VERSION_MINOR=${SENSEI_VERSION_MINOR}")
message(STATUS "SENSEI_VERSION_PATCH=${SENSEI_VERSION_PATCH}")

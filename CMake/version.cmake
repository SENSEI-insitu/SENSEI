set(tmp)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE}
        --git-dir=${CMAKE_SOURCE_DIR}/.git describe --tags
        OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT tmp)
    set(tmp "v0.0.0")
endif()
set(SENSEI_VERSION ${tmp} CACHE STRING "SENSEI version")

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)\\-[0-9]+\\-([0-9a-zA-Z]+)"
  "\\1" SENSEI_VERSION_MAJOR ${SENSEI_VERSION})

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)\\-[0-9]+\\-([0-9a-zA-Z]+)"
  "\\2" SENSEI_VERSION_MINOR ${SENSEI_VERSION})

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)\\-[0-9]+\\-([0-9a-zA-Z]+)"
  "\\3" SENSEI_VERSION_PATCH ${SENSEI_VERSION})

string(REGEX REPLACE "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)\\-[0-9]+\\-([0-9a-zA-Z]+)"
  "\\4" SENSEI_VERSION_DEVEL ${SENSEI_VERSION})

message(STATUS "SENSEI_VERSION_MAJOR=${SENSEI_VERSION_MAJOR}")
message(STATUS "SENSEI_VERSION_MINOR=${SENSEI_VERSION_MINOR}")
message(STATUS "SENSEI_VERSION_PATCH=${SENSEI_VERSION_PATCH}")

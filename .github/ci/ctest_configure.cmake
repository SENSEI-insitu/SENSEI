include("${CMAKE_CURRENT_LIST_DIR}/github_actions.cmake")

set(cmake_args
  -C "${CMAKE_CURRENT_LIST_DIR}/configure_$ENV{CMAKE_CONFIGURATION}.cmake")

# Create an entry in CDash.
ctest_start(${ctest_model} TRACK "${ctest_track}")

# Gather update information.
find_package(Git)
set(CTEST_UPDATE_VERSION_ONLY ON)
set(CTEST_UPDATE_COMMAND "${GIT_EXECUTABLE}")
ctest_update()

# Configure the project.
ctest_configure(
  OPTIONS "${cmake_args}"
  RETURN_VALUE configure_result)

# Read the files from the build directory.
ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")

# We can now submit because we've configured. This is a cmb-superbuild-ism.
if (DO_SUBMIT)
  ctest_submit(PARTS Update)
  ctest_submit(PARTS Configure)
endif ()

if (configure_result)
  message(FATAL_ERROR
    "Failed to configure")
endif ()

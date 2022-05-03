set(BUILD_TESTING ON CACHE BOOL "")
set(ENABLE_SENSEI ON CACHE BOOL "")
set(MPIEXEC_PREFLAGS "--allow-run-as-root" CACHE STRING "")
set(TEST_NP 2 CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/configure_options.cmake)

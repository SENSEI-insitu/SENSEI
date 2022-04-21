execute_process(
  COMMAND "which" "python3"
  OUTPUT_VARIABLE pyexec
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Python3_EXECUTABLE "${pyexec}" CACHE PATH "")
set(Python3_EXECUTABLE "${pyexec}" CACHE PATH "")

get_filename_component(pypath "${pyexec}" DIRECTORY)
get_filename_component(pypath "${pypath}" DIRECTORY)
set(Python3_ROOT_DIR "${pypath}" CACHE PATH "")

set(ENABLE_PYTHON ON CACHE BOOL "")
set(SENSEI_PYTHON_VERSION 3 CACHE STRING "")

if (ENABLE_CATALYST)
  set(ENABLE_CATALYST_PYTHON ON CACHE BOOL "")
endif ()

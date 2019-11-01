set(SENSEI_DATA_ROOT "" CACHE PATH "Path to SENSEI test data")
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
  enable_testing()
  include(CTest)
endif()

set(TEST_NP "4" CACHE STRING "Number of procs to use in parallel tests")
math(EXPR TEST_NP_HALF "${TEST_NP}/2")

# senseiAddTest(name
#   EXEC_NAME -- optional, name of the copiled test
#   SOURCES  -- optional, source files to comile
#   LIBS -- optional, libraries to link to the compiled test
#   COMMAND -- required, test command
#   FEATURES -- optional, boolean condition decribing feature dependencies
#   REQ_SENSEI_DATA -- flag whose presence indicates the test needs the data repo
#   )
function (senseiAddTest T_NAME)
  set(opt_args REQ_SENSEI_DATA)
  set(val_args EXEC_NAME)
  set(array_args SOURCES LIBS COMMAND FEATURES)
  cmake_parse_arguments(T "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
  set(TEST_ENABLED ON)
  if (NOT DEFINED T_FEATURES)
    set(TEST_ENABLED ON)
  else()
    foreach(feature ${T_FEATURES})
      if (NOT feature)
        set(TEST_ENABLED OFF)
      endif()
    endforeach()
  endif()
  if (TEST_ENABLED)
    if (T_SOURCES)
      set(EXEC_NAME ${T_NAME})
      if (T_EXEC_NAME)
        set(EXEC_NAME ${T_EXEC_NAME})
      endif()
      add_executable(${EXEC_NAME} ${T_SOURCES})
      if (T_LIBS)
        target_link_libraries(${EXEC_NAME} ${T_LIBS})
      endif()
    endif()
    if ((T_REQ_SENSEI_DATA AND SENSEI_DATA_ROOT) OR NOT T_REQ_SENSEI_DATA)
      add_test(NAME ${T_NAME} COMMAND ${T_COMMAND}
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      set_tests_properties(${T_NAME}
        PROPERTIES FAIL_REGULAR_EXPRESSION "[Ee][Rr][Rr][Oo][Rr]")
    endif()
  endif()
endfunction()

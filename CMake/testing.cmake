set(SENSEI_DATA_ROOT "" CACHE PATH "Path to SENSEI test data")
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
  enable_testing()
  include(CTest)
endif()

set(TEST_NP "4" CACHE STRING "Number of procs to use in parallel tests")
math(EXPR TEST_NP_HALF "${TEST_NP}/2")

#[==[.md
Add a test for the Sensei project.

~~~
senseiAddTest(name
  COMMAND <cmd>...
  [PARALLEL <N>]
  [PARALLEL_SHELL <N>]
  [SOURCES  <source>...
   [EXEC_NAME <name>]
   [LIBS <libraries>...]]
  [FEATURES <feature>...]
  [REQ_SENSEI_DATA]
  [PROPERTIES <property>...])
~~~

All options are optional unless otherwise specified.

  * `COMMAND`: (required) Base command used to run the test
  * `PARALLEL`:
    Execute this test using MPI with the specified number of ranks.
    Adds test label `PARALLEL`.
  * `PARALLEL_SHELL`:
    Export environment variables for MPI run from a shell script. Conflicts with option PARALLEL.
    Adds test labels `PARALLEL` and `SHELL`.
      `export MPI_PROCESSES=<N>`
      `export MPI_PROCESSES_HALF=<N/2>`
      `export MPI_LAUNCHER=mpiexec <preargs> -np <N>`
      `export MPIEXEC_POSTFLAGS=<MPIEXEC_POSTFLAGS>`
  * `SOURCES`: Source files to compile for the test executable
  * `EXEC_NAME `: Name of the compiled test executable (default: `<name>` passed to `senseAddTest`)
  * `LIBS`: Libraries to link to the compiled test executable
  * `FEATURES`: List of features that must be enabled for the test to run.
                Maps to SENSEI_ENABLE_<feature> and adds <feature> labels to the test.
  * `REQ_SENSEI_DATA`: Flag to indicate the test needs the data repo.
  * `PROPERTIES`: [Test  properties](https://cmake.org/cmake/help/v3.6/manual/cmake-properties.7.html\#test-properties) for this test
#]==]
function (senseiAddTest T_NAME)
  set(opt_args REQ_SENSEI_DATA CUDA_TARGET)
  set(val_args EXEC_NAME PARALLEL PARALLEL_SHELL)
  set(array_args SOURCES LIBS COMMAND FEATURES PROPERTIES)
  cmake_parse_arguments(PARSE_ARGV 0 T "${opt_args}" "${val_args}" "${array_args}")

  if (T_PARALLEL AND T_PARALLEL_SHELL)
    message(FATAL_ERROR "Test cannot be marked PARALLEL and PARALLEL_SHELL at the same time")
  endif ()

  if (NOT T_COMMAND)
    message(FATAL_ERROR
      "Test must have a command to run")
  endif ()

  # Check if the test should be enabled
  # based on the enabled features
  set(TEST_ENABLED ON)
  if (NOT DEFINED T_FEATURES)
    set(TEST_ENABLED ON)
  else()
    foreach(feature ${T_FEATURES})
      if (NOT SENSEI_ENABLE_${feature})
        set(TEST_ENABLED OFF)
      endif()
    endforeach()
  endif()

  if (TEST_ENABLED)
    # Build the executable if there are sources provided
    if (T_SOURCES)
      set(EXEC_NAME ${T_NAME})
      if (T_EXEC_NAME)
        set(EXEC_NAME ${T_EXEC_NAME})
      endif()
      add_executable(${EXEC_NAME} ${T_SOURCES})
      if (T_LIBS)
        target_link_libraries(${EXEC_NAME} ${T_LIBS})
      endif()
      if (SENSEI_ENABLE_CUDA AND T_CUDA_TARGET)
        sensei_cuda_target(TARGET ${EXEC_NAME} SOURCES ${T_SOURCES})
      endif ()
    endif()

    if ((T_REQ_SENSEI_DATA AND SENSEI_DATA_ROOT) OR NOT T_REQ_SENSEI_DATA)
      # Configure the tests COMMAND and properties
      set(test_env)
      if (T_FEATURES)
        set(test_labels ${T_FEATURES})
      else ()
        set(test_labels GENERAL)
      endif ()

      # Configure Parallel tests
      if (T_PARALLEL)
        set(mpi_command ${MPIEXEC};${MPIEXEC_NUMPROC_FLAG};${T_PARALLEL};${MPIEXEC_PREFLAGS})
        list(PREPEND T_COMMAND ${mpi_command})
        list(APPEND T_COMMAND ${MPIEXEC_POSTFLAGS})
        if (T_PARALLEL GREATER 1)
          list(APPEND test_labels PARALLEL)
        endif ()
      elseif (T_PARALLEL_SHELL)
        set(mpi_command ${MPIEXEC};${MPIEXEC_NUMPROC_FLAG};${T_PARALLEL_SHELL};${MPIEXEC_PREFLAGS})
        string(REPLACE ";" " " mpi_launcher "${mpi_command}")
        math(EXPR processors_half "${T_PARALLEL_SHELL}/2")
        list(APPEND test_env
          MPI_PROCESSES=${T_PARALLEL_SHELL}
          MPI_PROCESSES_HALF=${processors_half}
          MPI_LAUNCHER=${mpi_launcher}
          MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS})
        if (T_PARALLEL_SHELL GREATER 1)
          list(APPEND test_labels PARALLEL)
        endif ()
        list(APPEND test_labels SHELL)
      else ()
        list(APPEND test_labels SERIAL)
      endif ()

      # Add the test
      add_test(NAME ${T_NAME} COMMAND ${T_COMMAND}
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

      # Set the properties
      if (T_PROPERTIES)
        set_tests_properties(${T_NAME} PROPERTIES ${T_PROPERTIES})
      endif ()

      # Set the fail regex
      get_property(has_fail_regex TEST ${T_NAME} PROPERTY FAIL_REGULAR_EXPRESSION SET)
      if (NOT has_fail_regex)
        set_property(TEST ${T_NAME} PROPERTY FAIL_REGULAR_EXPRESSION "[Ee][Rr][Rr][Oo][Rr]")
      endif ()

      # Add extra environment variables
      set_property(TEST ${T_NAME} APPEND PROPERTY ENVIRONMENT ${test_env})
      if (SENSEI_PYTHON_SITE)
        set_property(TEST ${T_NAME} APPEND PROPERTY
          ENVIRONMENT_MODIFICATION
            "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_SITE}")
      endif ()
      # Add extra labels
      set_property(TEST ${T_NAME} APPEND PROPERTY LABELS ${test_labels})

      # Set the correct number of processes
      if (T_PARALLEL)
        set_property(TEST ${T_NAME} PROPERTY PROCESSORS ${T_PARALLEL})
      elseif (T_PARALLEL_SHELL)
        set_property(TEST ${T_NAME} PROPERTY PROCESSORS ${T_PARALLEL_SHELL})
      else ()
        set_property(TEST ${T_NAME} PROPERTY PROCESSORS 1)
      endif ()
    endif()
  endif()
endfunction()

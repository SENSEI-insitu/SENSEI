if (ENABLE_PYTHON)
  # TODO -- Python 3
  find_package(PythonInterp REQUIRED)
  if(PYTHONINTERP_FOUND)
      find_program(PYTHON_CONFIG_EXECUTABLE python-config)
      if (NOT PYTHON_CONFIG_EXECUTABLE)
          message(SEND_ERROR "python-config executable is required.")
      endif()
      execute_process(COMMAND ${PYTHON_CONFIG_EXECUTABLE} --prefix
          OUTPUT_VARIABLE python_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PYTHON_INCLUDE_DIR ${python_prefix}/include/python2.7)
      if (EXISTS ${python_prefix}/lib/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
          set(PYTHON_LIBRARY ${python_prefix}/lib/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
      elseif (EXISTS ${python_prefix}/lib64/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
          set(PYTHON_LIBRARY ${python_prefix}/lib64/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
      elseif (EXISTS ${python_prefix}/lib/x86_64-linux-gnu/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
          set(PYTHON_LIBRARY ${python_prefix}/lib/x86_64-linux-gnu/libpython2.7${CMAKE_SHARED_LIBRARY_SUFFIX})
      else()
          message(SEND_ERROR "Failed to locate Python library for ${python_prefix}")
      endif()
  endif()
  find_package(PythonLibs REQUIRED)
  find_package(NUMPY REQUIRED)
  find_program(swig_cmd NAMES swig swig3.0)
  if (swig_cmd-NOTFOUND)
  	message(SEND_ERROR "Failed to locate swig")
  endif()
  find_package(MPI4PY REQUIRED)

  add_library(sPython INTERFACE)
  target_include_directories(sPython INTERFACE ${PYTHON_INCLUDE_PATH} ${MPI4PY_INCLUDE_DIR})
  target_link_libraries(sPython INTERFACE ${PYTHON_LIBRARIES} sVTK)
  install(TARGETS sPython EXPORT sPython)
  install(EXPORT sPython DESTINATION lib/cmake EXPORT_LINK_INTERFACE_LIBRARIES)

  function(depend_swig input output)
      set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
      set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
      # custom command to update the dependency file
      add_custom_command(
          OUTPUT ${output_file}
          COMMAND ${swig_cmd} -c++ -python -MM
              -I${MPI4PY_INCLUDE_DIR} -I${CMAKE_BINARY_DIR}
              -I${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}
              -I${CMAKE_SOURCE_DIR}/sensei -I${CMAKE_SOURCE_DIR}/python
              ${input_file} | sed -e 's/[[:space:]\\]\\{1,\\}//g' -e '1,2d' -e '/senseiConfig\\.h/d' > ${output_file}
          MAIN_DEPENDENCY ${input_file}
          COMMENT "Generating dependency file for ${input}...")
      # bootstrap the dependency list
      message(STATUS "Generating initial dependency list for ${input}")
      execute_process(
          COMMAND ${swig_cmd} -c++ -python -MM
              -I${MPI4PY_INCLUDE_DIR} -I${CMAKE_BINARY_DIR}
              -I${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}
              -I${CMAKE_SOURCE_DIR}/sensei -I${CMAKE_SOURCE_DIR}/python
              -DSWIG_BOOTSTRAP ${input_file}
         COMMAND sed -e s/[[:space:]\\]\\{1,\\}//g -e 1,2d -e /senseiConfig\\.h/d
         OUTPUT_FILE ${output_file})
  endfunction()

  function(wrap_swig input output depend ttable)
      set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
      set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
      set(depend_file ${CMAKE_CURRENT_BINARY_DIR}/${depend})
      file(STRINGS ${depend_file} depends)
      add_custom_command(
          OUTPUT ${output_file}
          COMMAND ${swig_cmd} -c++ -python -w341,325,401,504
              -DSWIG_TYPE_TABLE=${ttable}
              -I${MPI4PY_INCLUDE_DIR} -I${CMAKE_BINARY_DIR}
              -I${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}
              -I${CMAKE_SOURCE_DIR}/sensei -I${CMAKE_SOURCE_DIR}/python
              -o ${output_file} ${input_file}
          MAIN_DEPENDENCY ${input_file}
          DEPENDS ${depend_file} ${depends}
          COMMENT "Generating python bindings for ${input}...")
  endfunction()
endif()

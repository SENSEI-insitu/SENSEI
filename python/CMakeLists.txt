if (ENABLE_PYTHON)
  depend_swig(senseiPython.i senseiPython.dep)
  wrap_swig(senseiPython.i senseiPython.cxx senseiPython.dep senseiPython)

  include_directories(SYSTEM ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

  PYTHON_ADD_MODULE(_senseiPython ${CMAKE_CURRENT_BINARY_DIR}/senseiPython.cxx)

  # include the directory containing _senseiPython in the rpath
  # this enables one not to have to set (DY)LD_LIBRARY_PATH
  # prior to importing sensseiPython
  if (APPLE)
    set_target_properties(_senseiPython PROPERTIES INSTALL_RPATH "@loader_path/./")
  elseif(UNIX)
    set_target_properties(_senseiPython PROPERTIES INSTALL_RPATH "\$ORIGIN/./")
  endif()

  target_link_libraries(_senseiPython PUBLIC sensei sPython)

  add_custom_command(TARGET _senseiPython POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/senseiPython.py
      ${CMAKE_CURRENT_BINARY_DIR}/../lib)

  configure_file(sensei.py ${CMAKE_CURRENT_BINARY_DIR}/../lib/sensei.py
    COPYONLY)

  install(TARGETS _senseiPython LIBRARY DESTINATION lib)
  install(FILES sensei.py ${CMAKE_CURRENT_BINARY_DIR}/senseiPython.py
    DESTINATION lib)
endif()

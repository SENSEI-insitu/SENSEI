if (ENABLE_PYTHON)

  # find Python
  set(SENSEI_PYTHON_VERSION 3 CACHE STRING
    "The major version number of Python SENSEI should use.")
  set_property(CACHE SENSEI_PYTHON_VERSION PROPERTY STRINGS 2 3)
  find_package(PythonInterp ${SENSEI_PYTHON_VERSION} REQUIRED)
  find_package(PythonLibs ${SENSEI_PYTHON_VERSION} REQUIRED)
  find_package(NUMPY REQUIRED)
  find_program(swig_cmd NAMES swig swig3.0 swig4.0)
  if (swig_cmd-NOTFOUND)
  	message(SEND_ERROR "Failed to locate swig")
  endif()

  # find MPI
  find_package(MPI4PY REQUIRED)

  # create the interface library
  add_library(sPython INTERFACE)
  target_include_directories(sPython INTERFACE ${PYTHON_INCLUDE_PATH} ${MPI4PY_INCLUDE_DIR})
  target_link_libraries(sPython INTERFACE ${PYTHON_LIBRARIES})
  install(TARGETS sPython EXPORT sPython)
  install(EXPORT sPython DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

  # find SWIG
  cmake_policy(SET CMP0078 NEW)
  cmake_policy(SET CMP0086 NEW)
  find_package(SWIG COMPONENTS python)
  include(UseSWIG)

  # the destination of all SENSEI Python codes
  set(SENSEI_PYTHON_DIR
    "${CMAKE_INSTALL_LIBDIR}/python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sensei/"
    CACHE STRING "Where SENSEI Python bindings are compiled and installed")

  message(STATUS "SENSEI Python modules will be installed at \"${SENSEI_PYTHON_DIR}\"")

endif()

if (ENABLE_PYTHON)

  # find Python
  set(SENSEI_PYTHON_VERSION 3 CACHE STRING
    "The major version number of Python SENSEI should use.")
  set_property(CACHE SENSEI_PYTHON_VERSION PROPERTY STRINGS 2 3)

  if (SENSEI_PYTHON_VERSION STREQUAL "3")
    find_package(Python3 COMPONENTS Interpreter Development)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(PYTHON_INCLUDE_PATH)
    set(PYTHON_LIBRARIES Python3::Python)
  elseif (SENSEI_PYTHON_VERSION STREQUAL "2")
    find_package(Python3 COMPONENTS Interpreter Development)
    set(PYTHON_EXECUTABLE ${Python2_EXECUTABLE})
    set(PYTHON_INCLUDE_PATH)
    set(PYTHON_LIBRARIES Python2::Python)
  else()
    message(FATAL_ERROR "Invalid Python Version: ${SENSEI_PYTHON_VERSION}")
  endif ()
  find_package(PythonInterp ${SENSEI_PYTHON_VERSION} REQUIRED)
  find_package(PythonLibs ${SENSEI_PYTHON_VERSION} REQUIRED)
  find_package(NUMPY REQUIRED)

  # find MPI
  find_package(MPI4PY REQUIRED)

  # create the interface library
  add_library(sPython INTERFACE)
  target_include_directories(sPython INTERFACE ${PYTHON_INCLUDE_PATH} ${MPI4PY_INCLUDE_DIR})
  target_link_libraries(sPython INTERFACE ${PYTHON_LIBRARIES})
  install(TARGETS sPython EXPORT sPython)
  install(EXPORT sPython DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES)

  # find SWIG
  cmake_policy(SET CMP0078 NEW)
  cmake_policy(SET CMP0086 NEW)
  find_package(SWIG REQUIRED COMPONENTS python)
  include(UseSWIG)

  # the destination of all SENSEI Python codes
  set(SENSEI_PYTHON_SITE
    "${CMAKE_INSTALL_LIBDIR}/python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/"
    CACHE STRING "Where Python modules are compiled and installed.")

  set(SENSEI_PYTHON_DIR "${SENSEI_PYTHON_SITE}/sensei/"
    CACHE STRING "Where SENSEI Python bindings are compiled and installed")

  message(STATUS "SENSEI Python modules will be installed at \"${SENSEI_PYTHON_DIR}\"")

endif()

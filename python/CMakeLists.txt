if (ENABLE_PYTHON)
    set_property(SOURCE senseiPython.i PROPERTY CPLUSPLUS ON)
    set_property(SOURCE senseiPython.i PROPERTY SWIG_MODULE_NAME senseiPython)
    set_property(SOURCE senseiPython.i PROPERTY DEPENDS ../sensei/DataAdaptor.i)

    swig_add_library(senseiPython TYPE MODULE
      LANGUAGE python SOURCES senseiPython.i
      OUTPUT_DIR "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}"
      OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    set_target_properties(senseiPython PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}")

    target_link_libraries(senseiPython PythonAnalysis sensei senseiCore pysvtk sPython)

    target_include_directories(senseiPython PUBLIC
      "${CMAKE_SOURCE_DIR}/python" "${CMAKE_SOURCE_DIR}/utils/pysvtk"
      "${NUMPY_INCLUDE_DIR}")

    set_property(TARGET senseiPython PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

    # install the SWIG'ed SENSEI module
    install(TARGETS senseiPython DESTINATION ${SENSEI_PYTHON_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/senseiPython.py
      DESTINATION ${SENSEI_PYTHON_DIR})

    # include the directory containing _senseiPython in the rpath this
    # enables one not to have to set (DY)LD_LIBRARY_PATH prior to importing
    # sensei.
    if (APPLE)
      set_target_properties(senseiPython PROPERTIES INSTALL_RPATH "@loader_path/./")
    elseif(UNIX)
      set_target_properties(senseiPython PROPERTIES INSTALL_RPATH "\$ORIGIN/;\$ORIGIN/../svtk/")
    endif()

  configure_file(sensei.py "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/__init__.py"
    COPYONLY)

  install(FILES  "${CMAKE_BINARY_DIR}/${SENSEI_PYTHON_DIR}/__init__.py"
    DESTINATION "${SENSEI_PYTHON_DIR}")

  # capture python path for use in automated CI
  file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/sensei-env.sh"
    CONTENT
[=[
#!/bin/bash
export PYTHONPATH=@CMAKE_BINARY_DIR@/@SENSEI_PYTHON_SITE@:$PYTHONPATH
export LD_LIBRARY_PATH=@CMAKE_BINARY_DIR@/@CMAKE_INSTALL_LIBDIR@:$LD_LIBTRARY_PATH
export DYLD_LIBRARY_PATH=@CMAKE_BINARY_DIR@/@CMAKE_INSTALL_LIBDIR@:$DYLD_LIBTRARY_PATH
]=]
    @ONLY)

endif()

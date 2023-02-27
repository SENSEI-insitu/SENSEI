if (ENABLE_PYTHON)
  ###########################################################################
  # SWIG Python bindings for SVTK
  ###########################################################################
  set_property(SOURCE svtk.i PROPERTY CPLUSPLUS ON)
  set_property(SOURCE svtk.i PROPERTY SWIG_MODULE_NAME pysvtk)
  #set_property(SOURCE svtk.i PROPERTY COMPILE_OPTIONS -Wextra)

  set(SVTK_PYTHON_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/svtk/"
    CACHE STRING "Where SVTK Python bindings are compiled and installed")

  message(STATUS "SVTK Python modules will be installed at \"${SVTK_PYTHON_DIR}\"")

  swig_add_library(pysvtk TYPE SHARED
    LANGUAGE python SOURCES svtk.i
    OUTPUT_DIR "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}"
    OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

  set_target_properties(pysvtk PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}"
    SUFFIX ".so")

  target_link_libraries(pysvtk SVTK::CommonCore SVTK::CommonDataModel sPython)

  set(pysvtk_INCLUDE_DIRS
    ${CMAKE_BINARY_DIR}/utils/SVTK/Common/Core
    ${CMAKE_SOURCE_DIR}/utils/SVTK/Common/Core
    ${CMAKE_BINARY_DIR}/utils/SVTK/Common/DataModel
    ${CMAKE_BINARY_DIR}/utils/SVTK/Utilities/KWIML
    ${CMAKE_SOURCE_DIR}/utils/SVTK/Utilities/KWIML
    ${CMAKE_BINARY_DIR}/utils/SVTK/Utilities/KWSys
    ${CMAKE_SOURCE_DIR}/utils/SVTK/Utilities/KWSys
    ${CMAKE_BINARY_DIR}/utils/SVTK/ThirdParty/utf8
    ${CMAKE_SOURCE_DIR}/utils/SVTK/ThirdParty/utf8
    ${CMAKE_SOURCE_DIR}/utils/SVTK/Utilities/KWIML/svtkkwiml/include/)

  target_include_directories(pysvtk
    PUBLIC ${Python_INCLUDE_DIRS} ${pysvtk_INCLUDE_DIRS})

  set_property(TARGET pysvtk PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

  # include the directory containing the module in the rpath this enables one
  # not to have to set (DY)LD_LIBRARY_PATH prior to importing svtk.
  if (APPLE)
    set_target_properties(pysvtk PROPERTIES INSTALL_RPATH "@loader_path/./")
  elseif(UNIX)
    set_target_properties(pysvtk PROPERTIES INSTALL_RPATH "\$ORIGIN/./")
  endif()

  # install the SWIG'ed SVTK package
  install(TARGETS pysvtk DESTINATION ${SVTK_PYTHON_DIR})

  install(FILES ${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}/pysvtk.py
    DESTINATION ${SVTK_PYTHON_DIR})

  #install(EXPORT pysvtk DESTINATION ${sensei_CMAKE_INSTALL_CMAKEDIR}
  #  EXPORT_LINK_INTERFACE_LIBRARIES)

  # install additional pythons sources
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}")

  set(python_files __init__.py numpy_support.py svtkConstants.py)
  foreach (python_file IN LISTS python_files)

    configure_file(${python_file}
      "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}/${python_file}"
      COPYONLY)

    install(FILES "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_DIR}/${python_file}"
      DESTINATION "${SVTK_PYTHON_DIR}")

  endforeach ()

endif()

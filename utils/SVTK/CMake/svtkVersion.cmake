# SVTK version number components.
set(SVTK_MAJOR_VERSION 9)
set(SVTK_MINOR_VERSION 0)
set(SVTK_BUILD_VERSION 1)

if (NOT SVTK_MINOR_VERSION LESS 100)
  message(FATAL_ERROR
    "The minor version number cannot exceed 100 without changing "
    "`SVTK_VERSION_CHECK`.")
endif ()

if (NOT SVTK_BUILD_VERSION LESS 100000000)
  message(FATAL_ERROR
    "The build version number cannot exceed 100000000 without changing "
    "`SVTK_VERSION_CHECK`.")
endif ()

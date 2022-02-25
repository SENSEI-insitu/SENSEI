In order to build this project just configure it with CMake as a
separate project.  After running the "Configure" step, there will be a
svtkWin32OutputWindowProcessEncoded.c at the top level of the build
tree.  There is no need to actually load and build the project with
Visual Studio.

This project is intended to generate
svtkWin32OutputWindowProcessEncoded.c for inclusion in the build of
svtkCommon.  The executable is self-deleting and is used by
svtkWin32ProcessOutputWindow.  It is an output window that runs as a
separate process and deletes its own executable on exit.  This is
useful so that if the main process crashes, the output window is still
usable, which is good since it probably explains the crash.

Currently the self-deletion mechanism works on all versions of windows
but only when compiled by a Visual Studio compiler in release mode.

If svtkWin32OutputWindowProcess.c can be implemented in a way that
works for all windows compilers, then this project can be integrated
into the main SVTK build process by adding a custom command to generate
svtkWin32OutputWindowProcessEncoded.c on the fly like this:

IF(WIN32)
  IF (NOT SVTK_USE_X)
    SET(SVTK_OWP_ENCODED_C
      ${SVTK_BINARY_DIR}/Common/svtkWin32OutputWindowProcessEncoded.c)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${SVTK_OWP_ENCODED_C}
      COMMAND ${CMAKE_COMMAND}
      ARGS -G\"${CMAKE_GENERATOR}\"
           -H${SVTK_SOURCE_DIR}/Utilities/OutputWindowProcess
           -B${SVTK_BINARY_DIR}/Utilities/OutputWindowProcess
           -DSVTK_OWP_OUTPUT=${SVTK_OWP_ENCODED_C}
      DEPENDS ${SVTK_SOURCE_DIR}/Utilities/OutputWindowProcess/svtkWin32OutputWindowProcess.c
      )
  ENDIF ()
ENDIF()

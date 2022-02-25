# This project is designed to be used as a custom command or be built
# manually.  See README.txt for build instructions.
IF(NOT SVTK_OWP_BUILD_INNER)
  PROJECT(SVTK_OWP)
  IF(NOT SVTK_OWP_OUTPUT)
    SET(SVTK_OWP_OUTPUT
      ${SVTK_OWP_BINARY_DIR}/svtkWin32OutputWindowProcessEncoded.c)
  ENDIF()
  IF(NOT SVTK_OWP_NAME)
    SET(SVTK_OWP_NAME Win32OutputWindowProcess)
  ENDIF()
  SET(CMAKE_TRY_COMPILE_CONFIGURATION Release)
  TRY_COMPILE(COMPILED
    ${SVTK_OWP_BINARY_DIR}/Inner
    ${SVTK_OWP_SOURCE_DIR}
    SVTK_OWP_INNER svtk_owp_encode
    CMAKE_FLAGS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_TRY_COMPILE_CONFIGURATION}
                -DSVTK_OWP_BUILD_INNER:BOOL=1
                -DSVTK_OWP_OUTPUT:STRING=${SVTK_OWP_OUTPUT}
                -DSVTK_OWP_NAME:STRING=${SVTK_OWP_NAME}
    OUTPUT_VARIABLE OUTPUT)
  IF(COMPILED)
    MESSAGE(STATUS "Creating \"${SVTK_OWP_OUTPUT}\" succeeded.")
  ELSE()
    FILE(WRITE ${SVTK_OWP_OUTPUT} "failed_to_generate\n")
    MESSAGE(FATAL_ERROR
      "Creating \"${SVTK_OWP_OUTPUT}\" failed with output [${OUTPUT}].")
  ENDIF()
ELSE()
  PROJECT(SVTK_OWP_INNER)

  ADD_EXECUTABLE(svtkWin32OutputWindowProcess WIN32 svtkWin32OutputWindowProcess.c)
  SET(SVTK_OWP_EXE svtkWin32OutputWindowProcess)
  GET_TARGET_PROPERTY(SVTK_OWP_EXE_PATH ${SVTK_OWP_EXE} LOCATION)

  IF(NOT BORLAND)
    TARGET_LINK_LIBRARIES(svtkWin32OutputWindowProcess gdi32)
  ENDIF()

  ADD_EXECUTABLE(svtkEncodeExecutable svtkEncodeExecutable.c)
  SET(SVTK_ENCODE_EXE svtkEncodeExecutable)
  ADD_CUSTOM_COMMAND(
    OUTPUT ${SVTK_OWP_OUTPUT}
    COMMAND ${SVTK_ENCODE_EXE}
    ARGS ${SVTK_OWP_EXE_PATH} ${SVTK_OWP_OUTPUT} ${SVTK_OWP_NAME}
    DEPENDS ${SVTK_OWP_EXE} ${SVTK_ENCODE_EXE}
    )

  ADD_CUSTOM_TARGET(svtk_owp_encode DEPENDS ${SVTK_OWP_OUTPUT})
ENDIF()

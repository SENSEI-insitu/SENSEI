# Check for vfw32 support
IF(NOT DEFINED SVTK_USE_VIDEO_FOR_WINDOWS)
  MESSAGE(STATUS "Checking if vfw32 is available")
  TRY_COMPILE(SVTK_USE_VIDEO_FOR_WINDOWS_DEFAULT
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
    ${CMAKE_CURRENT_LIST_DIR}/svtkTestvfw32.cxx
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=vfw32"
    OUTPUT_VARIABLE OUTPUT)
  IF(SVTK_USE_VIDEO_FOR_WINDOWS_DEFAULT)
    MESSAGE(STATUS "Checking if vfw32 is available -- yes")
    OPTION(SVTK_USE_VIDEO_FOR_WINDOWS "Enable using Video for Windows (vfw32) for video input and output." ON)
    FILE(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log
      "Checking if vfw32 is available "
      "passed with the following output:\n"
      "${OUTPUT}\n")
  ELSE()
    MESSAGE(STATUS "Checking if vfw32 is available -- no")
    OPTION(SVTK_USE_VIDEO_FOR_WINDOWS "Enable using Video for Windows (vfw32) for video input and output." OFF)
    FILE(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log
      "Checking if vfw32 is available "
      "failed with the following output:\n"
      "${OUTPUT}\n")
  ENDIF()
  MARK_AS_ADVANCED(SVTK_USE_VIDEO_FOR_WINDOWS)
ENDIF()

# Check if vfw32 supports the video capture functions
IF(SVTK_USE_VIDEO_FOR_WINDOWS)
  IF(NOT DEFINED SVTK_VFW_SUPPORTS_CAPTURE)
    MESSAGE(STATUS "Checking if vfw32 supports video capture")
    TRY_COMPILE(SVTK_VFW_SUPPORTS_CAPTURE
      ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
      ${CMAKE_CURRENT_LIST_DIR}/svtkTestvfw32Capture.cxx
      CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=vfw32"
      OUTPUT_VARIABLE OUTPUT)
    IF(SVTK_VFW_SUPPORTS_CAPTURE)
      MESSAGE(STATUS "Checking if vfw32 supports video capture -- yes")
      SET(SVTK_VFW_SUPPORTS_CAPTURE 1 CACHE INTERNAL "Enable using Video for Windows (vfw32) for video capture.")
      FILE(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log
        "Checking if vfw32 supports video capture "
        "passed with the following output:\n"
        "${OUTPUT}\n")
    ELSE()
      MESSAGE(STATUS "Checking if vfw32 supports video capture -- no")
      SET(SVTK_VFW_SUPPORTS_CAPTURE 0 CACHE INTERNAL "Enable using Video for Windows (vfw32) for video capture.")
      FILE(APPEND ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log
        "Checking if vfw32 supports video capture "
        "failed with the following output:\n"
        "${OUTPUT}\n")
    ENDIF()
  ENDIF()
ELSE()
  SET(SVTK_VFW_SUPPORTS_CAPTURE 0)
ENDIF()

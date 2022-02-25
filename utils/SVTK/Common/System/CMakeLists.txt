svtk_module_find_package(
  PACKAGE Threads)

set(classes
  svtkClientSocket
  svtkDirectory
  svtkServerSocket
  svtkSocket
  svtkSocketCollection
  svtkThreadMessager
  svtkTimerLog)

include("${CMAKE_CURRENT_SOURCE_DIR}/svtkTestSocketFeatures.cmake")

set(svtkSocket_defines)
if (SVTK_HAVE_GETSOCKNAME_WITH_SOCKLEN_T)
  list(APPEND svtkSocket_defines "SVTK_HAVE_GETSOCKNAME_WITH_SOCKLEN_T")
endif ()
if (SVTK_HAVE_SO_REUSEADDR)
  list(APPEND svtkSocket_defines "SVTK_HAVE_SO_REUSEADDR")
endif ()
if(SVTK_NO_PLATFORM_SOCKETS)
  list(APPEND svtkSocket_defines "SVTK_SOCKET_FAKE_API")
endif()

set_source_files_properties(svtkSocket.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${svtkSocket_defines}")

svtk_module_add_module(SVTK::CommonSystem
  CLASSES ${classes})
svtk_module_link(SVTK::CommonSystem
  PRIVATE
    $<$<PLATFORM_ID:WIN32>:wsock32>
    Threads::Threads)

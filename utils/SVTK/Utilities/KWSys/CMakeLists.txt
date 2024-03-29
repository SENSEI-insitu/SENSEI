#-----------------------------------------------------------------------------
# Configure KWSys to be named "svtksys".
SET(KWSYS_NAMESPACE svtksys)
SET(KWSYS_BUILD_PIC 1)
SET(KWSYS_ALIAS_TARGET SVTK::svtksys)
SET(KWSYS_USE_Base64 1)
SET(KWSYS_USE_CommandLineArguments 1)
SET(KWSYS_USE_DynamicLoader 1)
SET(KWSYS_USE_Process 1)
SET(KWSYS_USE_RegularExpression 1)
SET(KWSYS_USE_SystemTools 1)
SET(KWSYS_USE_SystemInformation 1)
SET(KWSYS_USE_MD5 1)
SET(KWSYS_USE_Glob 1)
set(KWSYS_HEADER_ROOT ${CMAKE_CURRENT_BINARY_DIR})
SET(KWSYS_INSTALL_COMPONENT_NAME_RUNTIME ${_svtk_build_TARGETS_COMPONENT})
SET(KWSYS_INSTALL_COMPONENT_NAME_DEVELOPMENT ${_svtk_build_HEADERS_COMPONENT})
SET(KWSYS_INSTALL_EXPORT_NAME ${_svtk_build_INSTALL_EXPORT})
set(KWSYS_SYSTEMTOOLS_SUPPORT_WINDOWS_SLASHES 1)
set(KWSYS_ENCODING_DEFAULT_CODEPAGE CP_UTF8)

if(NOT SVTK_INSTALL_NO_LIBRARIES)
  set(KWSYS_INSTALL_BIN_DIR ${_svtk_build_RUNTIME_DESTINATION})
  set(KWSYS_INSTALL_LIB_DIR ${_svtk_build_LIBRARY_DESTINATION})
endif()
if(_svtk_build_INSTALL_HEADERS)
  set(KWSYS_INSTALL_INCLUDE_DIR ${_svtk_build_HEADERS_DESTINATION})
endif()

svtk_module_third_party_internal(
  STANDARD_INCLUDE_DIRS
  LICENSE_FILES "svtksys/Copyright.txt"
  VERSION       "master")

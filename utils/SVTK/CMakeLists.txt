cmake_minimum_required(VERSION 3.8...3.12 FATAL_ERROR)
foreach(policy
    CMP0083 # CMake 3.14
    CMP0071 # CMake 3.10
    CMP0068 # CMake 3.9
    CMP0127
    )
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

if (POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif ()
if (POLICY CMP0065)
  cmake_policy(SET CMP0065 NEW)
endif ()

project(SVTK)

set(svtk_cmake_dir "${SVTK_SOURCE_DIR}/CMake")
list(INSERT CMAKE_MODULE_PATH 0 "${svtk_cmake_dir}")

if (APPLE)
  include(svtkApple)
endif ()

# must be before the following iOS / Android
include(svtkVersion)
set(SVTK_VERSION
    "${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}.${SVTK_BUILD_VERSION}")

option(SVTK_IOS_BUILD "Build svtk.framework for iOS" OFF)
mark_as_advanced(SVTK_IOS_BUILD)
if (SVTK_IOS_BUILD)
  include(svtkiOS)
  return()
endif()

option(SVTK_ANDROID_BUILD "Build SVTK for Android" OFF)
mark_as_advanced(SVTK_ANDROID_BUILD)
if (SVTK_ANDROID_BUILD)
  include(svtkAndroid)
  return()
endif()

include(svtkCMakeBackports)

if (SVTK_WHEEL_BUILD)
  include(svtkWheelPreparation)
endif ()

include(svtkCompilerChecks)
include(svtkCompilerPlatformFlags)
include(svtkCompilerExtraFlags)
include(svtkInitializeBuildType)
include(svtkSupportMacros)
include(svtkDownload)
include(svtkTesting)
include(svtkDirectories)
include(svtkMobileDevices)
include(svtkWrapSettings)
include(svtkCrossCompiling)
include(svtkObjectFactory)

set(CMAKE_INSTALL_LICENSEDIR ""
  CACHE PATH "License files (DATAROOTDIR/licenses/${CMAKE_PROJECT_NAME}")
mark_as_advanced(CMAKE_INSTALL_LICENSEDIR)
if (NOT CMAKE_INSTALL_LICENSEDIR)
  set(CMAKE_INSTALL_LICENSEDIR
    "${CMAKE_INSTALL_DATAROOTDIR}/licenses/${CMAKE_PROJECT_NAME}")
endif ()

# Setup compiler flags for dynamic analysis
# Should be included after svtkTesting
include(svtkCompilerDynamicAnalysisFlags)

set(SVTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-----------------------------------------------------------------------------
# Add the Remote Subdirectory
#add_subdirectory(Remote)

#-----------------------------------------------------------------------------
# Does SVTK require support for 64 bit file systems
include(CheckCXXSourceRuns)
file(READ "${svtk_cmake_dir}/svtkRequireLargeFilesSupport.cxx"
  SVTK_REQUIRE_LARGE_FILE_SUPPORT_FILE)
check_cxx_source_runs("${SVTK_REQUIRE_LARGE_FILE_SUPPORT_FILE}"
  CMAKE_REQUIRE_LARGE_FILE_SUPPORT "Support for 64 bit file systems")
set(SVTK_REQUIRE_LARGE_FILE_SUPPORT ${CMAKE_REQUIRE_LARGE_FILE_SUPPORT})

#-----------------------------------------------------------------------------
# Provide compatibility options.
option(SVTK_LEGACY_REMOVE "Remove all legacy code completely." OFF)
option(SVTK_LEGACY_SILENT "Silence all legacy code messages." OFF)
mark_as_advanced(SVTK_LEGACY_REMOVE SVTK_LEGACY_SILENT)

# Tell SVTK source files they are being built inside SVTK.
add_definitions(-DSVTK_IN_SVTK)

include(CheckTypeSize)

option(SVTK_USE_MPI "Support MPI" OFF)

set(SVTK_BUILD_TESTING "OFF"
  CACHE STRING "Build module testing directories")
set_property(CACHE SVTK_BUILD_TESTING
  PROPERTY
    STRINGS "ON;OFF;WANT")
set(BUILD_TESTING OFF)
if (SVTK_BUILD_TESTING)
  set(BUILD_TESTING ON)
endif ()

#-----------------------------------------------------------------------------
# Add an option to enable/disable components that have CUDA.
option(SVTK_USE_CUDA "Support CUDA compilation" OFF)
if(SVTK_USE_CUDA)
  # This needs to be a top level flag as enable_language needs
  # to be called from CMake entry point for variables to have
  # the correct scope.
  enable_language(CUDA)
endif()

#-----------------------------------------------------------------------------
# Add an option to enable/disable logging support.
option(SVTK_ENABLE_LOGGING "Enable logging support." ON)
mark_as_advanced(SVTK_ENABLE_LOGGING)

include(svtkEncodeString)

# Set up our directory structure for output libraries and binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SVTK_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SVTK_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${SVTK_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

set(svtk_requested_modules)
set(svtk_rejected_modules)
if (SVTK_WRAP_PYTHON)
  list(APPEND svtk_requested_modules
    SVTK::WrappingPythonCore
    SVTK::PythonInterpreter)
else ()
  list(APPEND svtk_rejected_modules
    SVTK::Python)
endif ()
if (SVTK_WRAP_JAVA)
  list(APPEND svtk_requested_modules
    SVTK::Java)
else ()
  list(APPEND svtk_rejected_modules
    SVTK::Java)
endif ()
# For svtkTkRenderWidget (only useful with Python, since WRAP_TCL is gone)
if (SVTK_USE_TK)
  set(SVTK_GROUP_ENABLE_Tk YES)
endif ()
if (SVTK_USE_MPI)
  set(SVTK_GROUP_ENABLE_MPI YES)
else ()
  # Reject `SVTK::mpi`, but allow those which happen to be part of the group to
  # be built.
  set(_svtk_module_group_default_MPI "DONT_WANT")
  list(APPEND svtk_rejected_modules
    SVTK::mpi)
endif ()
if (SVTK_ENABLE_LOGGING)
  list(APPEND svtk_requested_modules
    SVTK::loguru)
endif ()

if (SVTK_BUILD_TESTING)
  list(APPEND svtk_requested_modules
    # Required for just about all SVTK tests.
    SVTK::TestingCore)
endif ()

list(APPEND svtk_requested_modules
  # Required for just about everything.
  SVTK::CommonCore
  SVTK::CommonDataModel)

if (SVTK_ENABLE_WRAPPING)
  list(APPEND svtk_requested_modules
    # Required for wrapping with hierarchy files.
    SVTK::WrappingTools)
endif ()

if (svtk_requested_modules)
  list(REMOVE_DUPLICATES svtk_requested_modules)
endif ()
if (svtk_rejected_modules)
  list(REMOVE_DUPLICATES svtk_rejected_modules)
endif ()

set_property(GLOBAL
  PROPERTY
    "_svtk_module_autoinit_include" "\"svtkAutoInit.h\"")

option(SVTK_BUILD_ALL_MODULES "Build all modules by default" OFF)
mark_as_advanced(SVTK_BUILD_ALL_MODULES)

# Turn on these groups by default.
set(_svtk_module_group_default_StandAlone "WANT")
set(_svtk_module_group_default_Rendering "WANT")

set(svtk_enable_tests "${SVTK_BUILD_TESTING}")
if (SVTK_FORBID_DOWNLOADS)
  message(STATUS "Module testing disabled due to `SVTK_FORBID_DOWNLOADS` being set.")
  set(svtk_enable_tests "OFF")
endif ()

include(svtkModule)
include(svtkModuleDebugging)
set(svtk_source_directories
  "${CMAKE_CURRENT_SOURCE_DIR}/Accelerators"
  "${CMAKE_CURRENT_SOURCE_DIR}/Charts"
  "${CMAKE_CURRENT_SOURCE_DIR}/Common"
  "${CMAKE_CURRENT_SOURCE_DIR}/Domains"
  "${CMAKE_CURRENT_SOURCE_DIR}/Filters"
  "${CMAKE_CURRENT_SOURCE_DIR}/Geovis"
  "${CMAKE_CURRENT_SOURCE_DIR}/GUISupport"
  "${CMAKE_CURRENT_SOURCE_DIR}/Imaging"
  "${CMAKE_CURRENT_SOURCE_DIR}/Infovis"
  "${CMAKE_CURRENT_SOURCE_DIR}/Interaction"
  "${CMAKE_CURRENT_SOURCE_DIR}/IO"
  "${CMAKE_CURRENT_SOURCE_DIR}/Parallel"
  "${CMAKE_CURRENT_SOURCE_DIR}/Rendering"
  "${CMAKE_CURRENT_SOURCE_DIR}/Testing"
  "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty"
  "${CMAKE_CURRENT_SOURCE_DIR}/Utilities"
  "${CMAKE_CURRENT_SOURCE_DIR}/Views"
  "${CMAKE_CURRENT_SOURCE_DIR}/Web"
  "${CMAKE_CURRENT_SOURCE_DIR}/Wrapping")

set(remote_default ON)
if (SVTK_WHEEL_BUILD)
  set(remote_default OFF)
endif ()
option(SVTK_ENABLE_REMOTE_MODULES "Enable remote modules" ON)
mark_as_advanced(SVTK_ENABLE_REMOTE_MODULES)
if (SVTK_ENABLE_REMOTE_MODULES)
  list(APPEND svtk_source_directories
    "${CMAKE_CURRENT_SOURCE_DIR}/Remote")
endif ()

svtk_module_find_modules(svtk_module_files ${svtk_source_directories})
svtk_module_find_kits(svtk_kit_files ${svtk_source_directories})

# If we're building only the compile tools, limit what is done.
option(SVTK_BUILD_COMPILE_TOOLS_ONLY "Build only SVTK's compile tools" OFF)
mark_as_advanced(SVTK_BUILD_COMPILE_TOOLS_ONLY)
if (SVTK_BUILD_COMPILE_TOOLS_ONLY)
  set(svtk_requested_modules
    SVTKCompileTools::WrappingTools)
  set(svtk_rejected_modules)
  set(svtk_module_files
    "${CMAKE_CURRENT_SOURCE_DIR}/Wrapping/Tools/svtkcompiletools.module")
  set(svtk_kit_files)
endif ()

cmake_dependent_option(SVTK_INSTALL_SDK "Install SDK components" ON
  "NOT SVTK_BUILD_COMPILE_TOOLS_ONLY" ON)
mark_as_advanced(SVTK_INSTALL_SDK)

svtk_module_scan(
  MODULE_FILES        ${svtk_module_files}
  KIT_FILES           ${svtk_kit_files}
  REQUEST_MODULES     ${svtk_requested_modules}
  REJECT_MODULES      ${svtk_rejected_modules}
  PROVIDES_MODULES    svtk_modules
  PROVIDES_KITS       svtk_kits
  REQUIRES_MODULES    svtk_required_modules
  UNRECOGNIZED_MODULES svtk_unrecognized_modules
  WANT_BY_DEFAULT     "${SVTK_BUILD_ALL_MODULES}"
  ENABLE_TESTS        "${svtk_enable_tests}")

if (svtk_required_modules OR svtk_unrecognized_modules)
  message(FATAL_ERROR
    "The following modules were requested or required, but not found: "
    "${svtk_required_modules};${svtk_unrecognized_modules}.")
endif ()

if ("SVTK::Python" IN_LIST svtk_modules)
  include(svtkModuleWrapPython)
  if (NOT DEFINED SVTK_PYTHON_SITE_PACKAGES_SUFFIX)
    svtk_module_python_default_destination(SVTK_PYTHON_SITE_PACKAGES_SUFFIX)
  endif ()
endif ()

option(SVTK_ENABLE_KITS "Enable kits compilation" OFF)
mark_as_advanced(SVTK_ENABLE_KITS)

option(SVTK_USE_EXTERNAL "Use external copies of third party libraries by default" OFF)
mark_as_advanced(SVTK_USE_EXTERNAL)

option(SVTK_VERSIONED_INSTALL "Install artifacts with versioned names." ON)
mark_as_advanced(SVTK_VERSIONED_INSTALL)
set(svtk_version_suffix "")
if (SVTK_VERSIONED_INSTALL)
  set(svtk_version_suffix "-${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}")
endif ()

set(svtk_install_export SVTK)
if (NOT DEFINED svtk_cmake_destination)
  set(svtk_cmake_destination
    "${sensei_CMAKE_INSTALL_CMAKEDIR}/svtk${svtk_version_suffix}")
  set(svtk_target_package)
  if (SVTK_BUILD_COMPILE_TOOLS_ONLY)
    set(svtk_install_export SVTKCompileTools)
    set(svtk_cmake_destination
      "${sensei_CMAKE_INSTALL_CMAKEDIR}/svtkcompiletools${svtk_version_suffix}")
    set(svtk_target_package PACKAGE "${svtk_install_export}")
  endif ()
endif ()

set(SVTK_CUSTOM_LIBRARY_SUFFIX "<DEFAULT>"
  CACHE STRING "Custom library file name suffix (defaults to the version number)")
mark_as_advanced(SVTK_CUSTOM_LIBRARY_SUFFIX)
if (SVTK_CUSTOM_LIBRARY_SUFFIX STREQUAL "<DEFAULT>")
  if (SVTK_VERSIONED_INSTALL)
    set(SVTK_CUSTOM_LIBRARY_SUFFIX "${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}")
  else ()
    set(SVTK_CUSTOM_LIBRARY_SUFFIX "")
  endif ()
endif ()

svtk_module_build(
  MODULES             ${svtk_modules}
  KITS                ${svtk_kits}
  INSTALL_EXPORT      "${svtk_install_export}"
  ${svtk_target_package}
  HEADERS_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/svtk${svtk_version_suffix}"
  ${svtk_hierarchy_destination_args}
  CMAKE_DESTINATION   "${svtk_cmake_destination}"
  LICENSE_DESTINATION "${CMAKE_INSTALL_LICENSEDIR}"
  LIBRARY_NAME_SUFFIX "${SVTK_CUSTOM_LIBRARY_SUFFIX}"
  VERSION             "${SVTK_VERSION}"
  SOVERSION           "1"
  TEST_DATA_TARGET    SVTKData
  INSTALL_HEADERS     "${SVTK_INSTALL_SDK}"
  BUILD_WITH_KITS     "${SVTK_ENABLE_KITS}"
  USE_EXTERNAL        "${SVTK_USE_EXTERNAL}"
  ENABLE_WRAPPING     "${SVTK_ENABLE_WRAPPING}"
  TEST_INPUT_DATA_DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/Testing"
  TEST_OUTPUT_DATA_DIRECTORY  "${CMAKE_CURRENT_BINARY_DIR}/ExternalData/Testing")
include(svtkModuleJson)
svtk_module_json(
  MODULES ${svtk_modules}
  OUTPUT  "modules.json")
include(svtkModuleGraphviz)
svtk_module_graphviz(
  MODULES ${svtk_modules}
  KIT_CLUSTERS ON
  PRIVATE_DEPENDENCIES OFF
  OUTPUT  "modules.dot")

if (SVTK_WRAP_PYTHON)
  get_property(svtk_required_python_modules GLOBAL
    PROPERTY  svtk_required_python_modules)
  if (svtk_required_python_modules)
    list(REMOVE_DUPLICATES svtk_required_python_modules)
  endif ()
  string(REPLACE ";" "\n" svtk_required_python_modules "${svtk_required_python_modules}")
  file(WRITE "${CMAKE_BINARY_DIR}/requirements.txt"
    "${svtk_required_python_modules}\n")
  get_property(svtk_soabi GLOBAL
    PROPERTY  _svtk_python_soabi)
  svtk_module_wrap_python(
    MODULES         ${svtk_modules}
    INSTALL_EXPORT  SVTKPython
    PYTHON_PACKAGE  "svtkmodules"
    LIBRARY_DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    MODULE_DESTINATION  "${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}"
    CMAKE_DESTINATION   "${svtk_cmake_destination}"
    INSTALL_HEADERS     "${SVTK_INSTALL_SDK}"
    SOABI               "${svtk_soabi}"
    WRAPPED_MODULES svtk_python_wrapped_modules
    TARGET          SVTK::svtkpythonmodules)

  add_subdirectory(Wrapping/Python)

  export(
    EXPORT    SVTKPython
    NAMESPACE SVTK::
    FILE      "${CMAKE_BINARY_DIR}/${svtk_cmake_destination}/SVTKPython-targets.cmake")
  install(
    EXPORT      SVTKPython
    NAMESPACE   SVTK::
    FILE        SVTKPython-targets.cmake
    DESTINATION "${svtk_cmake_destination}"
    COMPONENT   "development")
endif ()

if (SVTK_WRAP_JAVA)
  enable_language(Java)

  add_subdirectory(Wrapping/Java)
endif ()

ExternalData_Add_Target(SVTKData)

# Create target to download data from the SVTKData group.  This must come after
# all tests have been added that reference the group, so we put it last.
if(SVTK_DATA_EXCLUDE_FROM_ALL)
  set_property(TARGET SVTKData PROPERTY EXCLUDE_FROM_ALL 1)
  if(BUILD_TESTING AND NOT SVTK_DATA_EXCLUDE_FROM_ALL_NO_WARNING)
    message(WARNING "SVTK_DATA_EXCLUDE_FROM_ALL is ON so test data "
      "(needed because BUILD_TESTING is ON) may not be available "
      "without manually building the 'SVTKData' target.")
  endif()
endif()

if (SVTK_INSTALL_SDK)
  set(svtk_cmake_build_dir
    "${CMAKE_CURRENT_BINARY_DIR}/${svtk_cmake_destination}")
  if (NOT SVTK_BUILD_COMPILE_TOOLS_ONLY)
    include(svtkInstallCMakePackage)
  else ()
    include(svtkInstallCMakePackageCompileTools)
  endif ()
endif ()

cmake_dependent_option(SVTK_ENABLE_CDASH_THIRD_PARTY_WARNINGS
  "Suppress warnings from third-party libraries from showing on CDash" OFF
  "SVTK_BUILD_TESTING" OFF)
mark_as_advanced(SVTK_ENABLE_CDASH_THIRD_PARTY_WARNINGS)

configure_file(
  "${svtk_cmake_dir}/CTestCustom.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake"
  @ONLY)

option(SVTK_BUILD_EXAMPLES "Build SVTK examples." OFF)
if (SVTK_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif ()

if(BUILD_TESTING)
  add_subdirectory(Testing/Install)
endif()

# The doxygen documentation needs to be aware of all modules.
option(SVTK_BUILD_DOCUMENTATION "Build the SVTK documentation" OFF)
if(SVTK_BUILD_DOCUMENTATION)
  add_subdirectory(Utilities/Doxygen)
endif()

# If python wrapping and testing is enabled then add driver scripts to run
# tests.  Note: Many python tests used to be automatically converted from TCL
# scripts. Hence the name svtkTclTest2Py
if(BUILD_TESTING AND SVTK_WRAP_PYTHON)
  add_subdirectory(Utilities/svtkTclTest2Py)
endif()

if (NOT TARGET uninstall)
  add_custom_target(uninstall
    COMMAND
      "${CMAKE_COMMAND}"
        "-DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        -P "${CMAKE_CURRENT_LIST_DIR}/CMake/svtkUninstall.cmake"
    USES_TERMINAL
    COMMENT "Uninstalling SVTK")
endif ()

install(
  FILES       "${CMAKE_CURRENT_LIST_DIR}/Copyright.txt"
  DESTINATION "${CMAKE_INSTALL_LICENSEDIR}"
  COMPONENT   "license")

# TODO: HeaderTest exclusions for memcheck.

if (SVTK_WHEEL_BUILD)
  include(svtkWheelFinalization)
endif ()

#-----------------------------------------------------------------------------
# Do this at the end so the all variables it uses are setup.
include(svtkBuildPath)

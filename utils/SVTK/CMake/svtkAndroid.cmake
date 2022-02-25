cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#
# Instructions:
# 1. Download and install the Android NDK.
# 2. Run ccmake -DSVTK_ANDROID_BUILD=ON /path/to/svtk/source
# 3. Set ANDROID_NDK to be the path to the NDK. (/opt/android-ndk by default)
# 4. Set API level and architecture.
# 5. Generate and make
#
include(ExternalProject)

# Convenience variables
set(PREFIX_DIR ${CMAKE_BINARY_DIR}/CMakeExternals/Prefix)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/CMakeExternals/Build)
set(INSTALL_DIR ${CMAKE_BINARY_DIR}/CMakeExternals/Install)

# Android options
set (_ANDROID_NDK_DEFAULT "/opt/android-ndk")
if (DEFINED ENV{ANDROID_NDK})
  set (_ANDROID_NDK_DEFAULT "$ENV{ANDROID_NDK}")
endif()
set(ANDROID_NDK ${_ANDROID_NDK_DEFAULT} CACHE PATH
  "Set to the absolute path of the Android NDK root directory.\
 A \$\{ANDROID_NDK\}/platforms directory must exist."
  )
if (NOT EXISTS "${ANDROID_NDK}/platforms")
  message(FATAL_ERROR "Please set a valid ANDROID_NDK path")
endif()
set(ANDROID_NATIVE_API_LEVEL "21" CACHE STRING "Android Native API Level")
set(ANDROID_ARCH_ABI "armeabi" CACHE STRING "Target Android architecture/abi")

# find android
set(example_flags)
if (SVTK_BUILD_EXAMPLES)
  find_program(ANDROID_EXECUTABLE
    NAMES android
    DOC   "The android command-line tool")
  if(NOT ANDROID_EXECUTABLE)
    message(FATAL_ERROR "Can not find android command line tool: android")
  endif()

  #find ant
  find_program(ANT_EXECUTABLE
    NAMES ant
    DOC   "The ant build tool")
  if(NOT ANT_EXECUTABLE)
    message(FATAL_ERROR "Can not find ant build tool: ant")
  endif()

  list(APPEND example_flags
    -DANDROID_EXECUTABLE:FILE=${ANDROID_EXECUTABLE}
    -DANT_EXECUTABLE:FILE=${ANT_EXECUTABLE}
  )
endif()

# Fail if the install path is invalid
if (NOT EXISTS ${CMAKE_INSTALL_PREFIX})
  message(FATAL_ERROR
    "Install path ${CMAKE_INSTALL_PREFIX} does not exist.")
endif()

# make sure we have a CTestCustom.cmake file
configure_file("${svtk_cmake_dir}/CTestCustom.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake" @ONLY)

# Compile a minimal SVTK for its compile tools
macro(compile_svtk_tools)
  ExternalProject_Add(
    svtk-compile-tools
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    PREFIX ${CMAKE_BINARY_DIR}/CompileTools
    BINARY_DIR ${CMAKE_BINARY_DIR}/CompileTools
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
    CMAKE_CACHE_ARGS
      -DSVTK_BUILD_COMPILE_TOOLS_ONLY:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DSVTK_BUILD_ALL_MODULES:BOOL=OFF
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DSVTK_BUILD_EXAMPLES:BOOL=OFF
      -DSVTK_BUILD_TESTING:BOOL=OFF
      -DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}
      -DSVTK_ENABLE_LOGGING:BOOL=OFF
  )
endmacro()
compile_svtk_tools()

# Hide some CMake configs from the user
mark_as_advanced(
  BUILD_SHARED_LIBS
  CMAKE_INSTALL_PREFIX
  CMAKE_OSX_ARCHITECTURES
  CMAKE_OSX_DEPLOYMENT_TARGET
)

# Now cross-compile SVTK with the android toolchain
set(android_cmake_flags
  ${example_flags}
  -DBUILD_SHARED_LIBS:BOOL=OFF
  -DSVTK_BUILD_TESTING:STRING=OFF
  -DSVTK_BUILD_EXAMPLES:BOOL=${SVTK_BUILD_EXAMPLES}
  -DSVTK_ENABLE_LOGGING:BOOL=OFF
  -DSVTK_GROUP_ENABLE_Rendering:STRING=DONT_WANT
  -DSVTK_GROUP_ENABLE_StandAlone:STRING=DONT_WANT
  -DSVTK_MODULE_ENABLE_SVTK_FiltersCore:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_FiltersModeling:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_FiltersSources:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_FiltersGeometry:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_IOGeometry:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_IOLegacy:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_IOImage:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_IOPLY:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_IOInfovis:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_ImagingCore:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_InteractionStyle:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_ParallelCore:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_RenderingCore:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_RenderingFreeType:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_TestingCore:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_TestingRendering:STRING=YES
  -DSVTK_MODULE_ENABLE_SVTK_RenderingVolumeOpenGL2:STRING=YES
)

if (SVTK_LEGACY_REMOVE)
  list(APPEND android_cmake_flags -DSVTK_LEGACY_REMOVE:BOOL=ON)
endif()

macro(crosscompile target api abi out_build_dir)
  set(_ANDROID_API "${api}")
  set(_ANDROID_ABI "${abi}")
  set(_ANDROID_DIR "${target}-${api}-${abi}")
  set(_ANDROID_TOOLCHAIN ${BUILD_DIR}/${_ANDROID_DIR}-toolchain.cmake)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/svtkAndroid-toolchain.cmake.in
    ${_ANDROID_TOOLCHAIN} @ONLY)
  ExternalProject_Add(
    ${target}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    PREFIX ${PREFIX_DIR}/${_ANDROID_DIR}
    BINARY_DIR ${BUILD_DIR}/${_ANDROID_DIR}
    INSTALL_DIR ${INSTALL_DIR}/${_ANDROID_DIR}
    DEPENDS svtk-compile-tools
    BUILD_ALWAYS 1
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}/${target}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_TOOLCHAIN_FILE:PATH=${_ANDROID_TOOLCHAIN}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DSVTKCompileTools_DIR:PATH=${CMAKE_BINARY_DIR}/CompileTools
      -DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}
      ${android_cmake_flags}
  )
  set(${out_build_dir} "${BUILD_DIR}/${_ANDROID_DIR}")
endmacro()
crosscompile(svtk-android "${ANDROID_NATIVE_API_LEVEL}" "${ANDROID_ARCH_ABI}" svtk_android_build_dir)

# Having issues getting the test to run after some
# changes on the device we use for testing
#
# add_test(NAME AndroidNative
#     WORKING_DIRECTORY ${svtk_android_build_dir}/Examples/Android/NativeSVTK/bin
#     COMMAND ${CMAKE_COMMAND}
#     -DWORKINGDIR=${svtk_android_build_dir}/Examples/Android/NativeSVTK/bin
#     -P ${CMAKE_CURRENT_SOURCE_DIR}/Examples/Android/NativeSVTK/runtest.cmake
#   )

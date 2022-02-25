# Force some SVTK options for wheels.
set(SVTK_PYTHON_VERSION 3)
set(SVTK_BUILD_TESTING OFF)
set(SVTK_ENABLE_WRAPPING ON)
set(SVTK_WRAP_PYTHON ON)
set(Python3_ARTIFACTS_INTERACTIVE ON)

if (UNIX AND NOT APPLE)
  # On Linux, prefer Legacy OpenGL library. We will revisit
  # this when all distributions will provides GLVND libraries by
  # default.
  if (NOT DEFINED OpenGL_GL_PREFERENCE OR OpenGL_GL_PREFERENCE STREQUAL "")
    set(OpenGL_GL_PREFERENCE "LEGACY")
  endif ()
endif ()

find_package(Python3 COMPONENTS Interpreter Development.Module)
set_property(GLOBAL PROPERTY _svtk_python_soabi "${Python3_SOABI}")

execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          -c "from distutils import util; print(util.get_platform())"
  OUTPUT_VARIABLE python_platform
  ERROR_VARIABLE  err
  RESULT_VARIABLE res
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE)
if (res)
  message(FATAL_ERROR
    "Failed to determine platform for Python implementation: ${err}")
endif ()

set(wheel_data_dir
  "svtk-${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}.${SVTK_BUILD_VERSION}.data")

# Set up the install tree for a wheel.
set(CMAKE_INSTALL_INCLUDEDIR
  "${wheel_data_dir}/headers")
set(svtk_cmake_destination
  "${wheel_data_dir}/headers/cmake")
set(CMAKE_INSTALL_SHAREDIR
  "${wheel_data_dir}/headers/cmake")
set(CMAKE_INSTALL_DATAROOTDIR
  "${wheel_data_dir}/share")
set(CMAKE_INSTALL_DOCDIR
  "${wheel_data_dir}/share")
set(svtk_hierarchy_destination_args
  HIERARCHY_DESTINATION "${wheel_data_dir}/headers/hierarchy")
set(setup_py_build_dir
  "build/lib.${python_platform}-${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
# Required for Windows DLL placement.
set(CMAKE_INSTALL_BINDIR
  # Must correlate with `svtk_module_wrap_python(PYTHON_PACKAGE)` argument
  "${setup_py_build_dir}/svtkmodules")
set(CMAKE_INSTALL_LIBDIR
  # Must correlate with `svtk_module_wrap_python(PYTHON_PACKAGE)` argument
  "${setup_py_build_dir}/svtkmodules")
set(SVTK_PYTHON_SITE_PACKAGES_SUFFIX ".")
set(SVTK_CUSTOM_LIBRARY_SUFFIX "")
set(SVTK_INSTALL_SDK OFF)
set(SVTK_INSTALL_PYTHON_EXES OFF)
set(BUILD_SHARED_LIBS ON)

if (APPLE)
  # macOS loader settings.
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_NAME_DIR "@rpath")
  list(APPEND CMAKE_INSTALL_RPATH
    "@loader_path")
elseif (UNIX)
  # ELF loader settings.
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  list(APPEND CMAKE_INSTALL_RPATH
    "$ORIGIN")
endif ()
set(SVTK_PYTHON_OPTIONAL_LINK ON)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/setup.py.in"
  "${CMAKE_BINARY_DIR}/setup.py"
  @ONLY)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/MANIFEST.in.in"
  "${CMAKE_BINARY_DIR}/MANIFEST.in"
  @ONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/Copyright.txt"
  "${CMAKE_BINARY_DIR}/LICENSE"
  COPYONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/README.md"
  "${CMAKE_BINARY_DIR}/README.md"
  COPYONLY)

unset(license_file)
unset(wheel_data_dir)

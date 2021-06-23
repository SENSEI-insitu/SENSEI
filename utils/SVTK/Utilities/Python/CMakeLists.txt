#[==[.md
# Python support

This module provides a single target for using Python within SVTK modules.

## Flexible Python libraries

Ideally, the libraries linked against the `svtkPython` module would be flexible
in the library actually used and it would be provided by the executable loading
the Python instead. This can be made to work with hacks currently, but the
following should be fixed first:

  - A better solution than `-undefined dynamic_lookup` for macOS. SVTK has
    [an issue][SVTK dynamic Python lookup issue] filed for this already.

[SVTK dynamic Python lookup issue]: https://gitlab.kitware.com/svtk/svtk/issues/17214
#]==]

if (NOT DEFINED SVTK_PYTHON_VERSION)
  set(SVTK_PYTHON_VERSION "3"
    CACHE STRING "")
  set_property(CACHE SVTK_PYTHON_VERSION
    PROPERTY
      STRINGS "2;3")
endif ()

# SVTK only supports a single Python version at a time, so make artifact finding
# interactive.
set("Python${SVTK_PYTHON_VERSION}_ARTIFACTS_INTERACTIVE" ON)

if (SVTK_PYTHON_VERSION STREQUAL "2")
  set(svtk_python_min_version "2.6")
  set(svtk_python_version_support "2.7")
  if (SVTK_REMOVE_LEGACY)
    set(svtk_python_min_version "${svtk_python_version_support}")
  endif ()
  svtk_module_find_package(
    PACKAGE Python2
    VERSION "${svtk_python_min_version}"
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed
    FORWARD_VERSION_REQ MINOR)
elseif (SVTK_PYTHON_VERSION STREQUAL "3")
  set(svtk_python_min_version "3.2")
  set(svtk_python_version_support "3.3")
  if (SVTK_REMOVE_LEGACY)
    set(svtk_python_min_version "${svtk_python_version_support}")
  endif ()
  svtk_module_find_package(
    PACKAGE Python3
    VERSION "${svtk_python_min_version}"
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed
    FORWARD_VERSION_REQ MINOR)
else ()
  message(FATAL_ERROR
    "`SVTK_PYTHON_VERSION` must either be 2 or 3.")
endif ()
set(svtk_python_includes "${Python${SVTK_PYTHON_VERSION}_INCLUDE_DIRS}")
set(svtk_python_target "Python${SVTK_PYTHON_VERSION}::Module")
set(svtk_python_embed_target "Python${SVTK_PYTHON_VERSION}::Python")
set(svtk_python_version "${Python${SVTK_PYTHON_VERSION}_VERSION}")
set(svtk_python_version_pair "${Python${SVTK_PYTHON_VERSION}_VERSION_MAJOR}.${Python${SVTK_PYTHON_VERSION}_VERSION_MINOR}")

set_property(GLOBAL PROPERTY _svtk_python_soabi "${Python${SVTK_PYTHON_VERSION}_SOABI}")

# Check deprecated versions of Python
if (NOT SVTK_LEGACY_SILENT AND svtk_python_version VERSION_LESS svtk_python_version_support)
  message(DEPRECATION
    "Python ${svtk_python_version} support is deprecated, use Python ${svtk_python_version_support}+")
endif ()

# Export location of python module dirs in install and build tree for every svtkpython module to use
# As long as those modules depend on svtkpython, they can retrieve and use these
if (NOT SVTK_PYTHON_SITE_PACKAGES_SUFFIX)
  if (WIN32 AND NOT CYGWIN)
    set(SVTK_PYTHON_SITE_PACKAGES_SUFFIX "Lib/site-packages")
  else ()
    set(SVTK_PYTHON_SITE_PACKAGES_SUFFIX
      "python${svtk_python_version_pair}/site-packages")
  endif ()
endif ()

if (CMAKE_CONFIGURATION_TYPES)
  # For build systems with configuration types e.g. Xcode/Visual Studio,
  # we rely on generator expressions.
  set(SVTK_BUILD_PYTHON_MODULES_DIR
    "${CMAKE_BINARY_DIR}/$<CONFIG>/${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}")
else ()
  set(SVTK_BUILD_PYTHON_MODULES_DIR
    "${CMAKE_BINARY_DIR}/${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}")
endif ()

if (NOT DEFINED SVTK_INSTALL_PYTHON_MODULES_DIR)
  if (WIN32 AND NOT CYGWIN)
    set(SVTK_INSTALL_PYTHON_MODULES_DIR
      "${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}")
  else ()
    set(SVTK_INSTALL_PYTHON_MODULES_DIR
      "${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}")
  endif ()
endif ()

# For static builds, since svtkPythonInterpreter cannot work off the library
# location, but instead has to use the executable location, it needs to know the
# library dir explicitly.
if (NOT SVTK_BUILD_SHARED_LIBS)
  set(SVTK_PYTHON_SITE_PACKAGES_SUFFIX_FIXED "${SVTK_INSTALL_PYTHON_MODULES_DIR}")
else ()
  set(SVTK_PYTHON_SITE_PACKAGES_SUFFIX_FIXED "${SVTK_PYTHON_SITE_PACKAGES_SUFFIX}")
endif ()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkPythonConfigure.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkPythonConfigure.h")

set(headers
  svtkPython.h
  "${CMAKE_CURRENT_BINARY_DIR}/svtkPythonConfigure.h")

svtk_module_add_module(SVTK::Python
  HEADERS ${headers}
  HEADER_ONLY)

include(CMakeDependentOption)
cmake_dependent_option(SVTK_PYTHON_OPTIONAL_LINK
  # Default to `ON` if it is possible.
  "Whether to link libpython from libraries or not" ON
  # We shouldn't do it for static builds and we can't do it without
  # `target_link_options`. Windows also always needs to link against libpython
  # directly.
  "NOT WIN32;BUILD_SHARED_LIBS;COMMAND target_link_options" OFF)

add_library(PythonUsed INTERFACE)
add_library(SVTK::PythonUsed ALIAS PythonUsed)
_svtk_module_install(PythonUsed)

# If we want optional linking and we have a real libpython, set up the forward
# linking.
if (SVTK_PYTHON_OPTIONAL_LINK AND TARGET "${svtk_python_embed_target}")
  # It has been observed that Ubuntu's GCC toolchain defaults require flags,
  # but are not detected here due to the confluence of the flags (basically,
  # the errors only show up in an executable that links to Python-using code,
  # but does not use Python itself either). Since this is not detected, setting
  # `svtk_undefined_symbols_allowed=OFF` manually may be required when using
  # Ubuntu toolchains.
  include("${CMAKE_CURRENT_SOURCE_DIR}/svtkUndefinedSymbolsAllowed.cmake")
  set(is_exe "$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>")
  set(needs_flags "$<NOT:$<BOOL:${svtk_undefined_symbols_allowed}>>")
  set(direct_link "$<BOOL:$<TARGET_PROPERTY:INTERFACE_svtk_python_direct_link>>")
  set(should_use "$<OR:${is_exe},${direct_link}>")
  set(platform_flags
    # Apple flags.
    "$<$<PLATFORM_ID:Darwin>:-undefined;dynamic_lookup>"
    # Linux flags.
    # The linker is allowed to skip libraries not used by the end result. This
    # can exclude the `libpython` DT_NEEDED entry at the final executable if it
    # itself does not actually use any libpython symbols.
    "$<$<AND:${is_exe},$<PLATFORM_ID:Linux>>:LINKER:--no-as-needed>"
    # Due to the above making the symbols actually available, we can ignore
    # unresolved symbols in the shared libraries that are being linked to.
    "$<$<AND:${is_exe},$<PLATFORM_ID:Linux>>:LINKER:--unresolved-symbols=ignore-in-shared-libs>")

  # Use the `PythonX::Module` target from FindPython.
  svtk_module_link(SVTK::Python
    INTERFACE
      "${svtk_python_target}")

  # This target is intended for use by the end-result executable with a
  # `SVTK::Python` link in its total library closure. This exposes the
  # `PythonX::Python` link at the executable level with the required platform
  # flags.
  target_link_libraries(PythonUsed
    INTERFACE
      "$<LINK_ONLY:$<${is_exe}:${svtk_python_embed_target}>>")
  target_link_options(PythonUsed
    INTERFACE
      "$<$<AND:${is_exe},${needs_flags}>:${platform_flags}>")
  _svtk_module_set_module_property(SVTK::Python APPEND
    PROPERTY  "forward_link"
    VALUE     SVTK::PythonUsed)
# Otherwise, if we have the libpython and aren't doing optional linking, just
# use libpython.
elseif (NOT SVTK_PYTHON_OPTIONAL_LINK AND TARGET "${svtk_python_embed_target}")
  svtk_module_link(SVTK::Python
    INTERFACE
      "${svtk_python_embed_target}")
# Do the optional linking support.
else ()
  svtk_module_link(SVTK::Python
    INTERFACE
      "${svtk_python_target}")
endif ()

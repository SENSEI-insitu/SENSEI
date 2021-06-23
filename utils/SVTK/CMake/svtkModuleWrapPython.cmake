#[==[
@defgroup module-wrapping-python Module Python CMake APIs
#]==]

#[==[
@file svtkModuleWrapPython.cmake
@brief APIs for wrapping modules for Python

@section Limitations

Known limitations include:

  - Shared Python modules only really support shared builds of modules. SVTK
    does not provide mangling facilities for itself, so statically linking SVTK
    into its Python modules precludes using SVTK's C++ interface anywhere else
    within the Python environment.
  - Only supports CPython. Other implementations are not supported by the
    `SVTK::WrapPython` executable.
  - Links directly to a Python library. See the `SVTK::Python` module for more
    details.
#]==]

#[==[
@ingroup module-wrapping-python
@brief Determine Python module destination

Some projects may need to know where Python expects its modules to be placed in
the install tree (assuming a shared prefix). This function computes the default
and sets the passed variable to the value in the calling scope.

~~~
svtk_module_python_default_destination(<var>
  [MAJOR_VERSION <major>])
~~~

By default, the destination is `${CMAKE_INSTALL_BINDIR}/Lib/site-packages` on
Windows and `${CMAKE_INSTALL_LIBDIR}/python<VERSION>/site-packages` otherwise.

`<MAJOR_VERSION>` must be one of `2` or `3`. If not specified, it defaults to
the value of `${SVTK_PYTHON_VERSION}`.
#]==]
function (svtk_module_python_default_destination var)
  cmake_parse_arguments(_svtk_module_python
    ""
    "MAJOR_VERSION"
    ""
    ${ARGN})

  if (_svtk_module_python_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_python_default_destination: "
      "${_svtk_module_python_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT _svtk_module_python_MAJOR_VERSION)
    if (NOT DEFINED SVTK_PYTHON_VERSION)
      message(FATAL_ERROR
        "A major version of Python must be specified (or `SVTK_PYTHON_VERSION` "
        "be set).")
    endif ()

    set(_svtk_module_python_MAJOR_VERSION "${SVTK_PYTHON_VERSION}")
  endif ()

  if (NOT _svtk_module_python_MAJOR_VERSION STREQUAL "2" AND
      NOT _svtk_module_python_MAJOR_VERSION STREQUAL "3")
    message(FATAL_ERROR
      "Only Python2 and Python3 are supported right now.")
  endif ()

  if (WIN32 AND NOT CYGWIN)
    set(destination "${CMAKE_INSTALL_BINDIR}/Lib/site-packages")
  else ()
    if (NOT DEFINED "Python${_svtk_module_python_MAJOR_VERSION}_VERSION_MAJOR" OR
        NOT DEFINED "Python${_svtk_module_python_MAJOR_VERSION}_VERSION_MINOR")
      find_package("Python${_svtk_module_python_MAJOR_VERSION}" QUIET COMPONENTS Development.Module)
    endif ()

    if (Python${_svtk_module_python_MAJOR_VERSION}_VERSION_MAJOR AND Python${_svtk_module_python_MAJOR_VERSION}_VERSION_MINOR)
      set(_svtk_python_version_suffix "${Python${SVTK_PYTHON_VERSION}_VERSION_MAJOR}.${Python${SVTK_PYTHON_VERSION}_VERSION_MINOR}")
    else ()
      message(WARNING
        "The version of Python is unknown; not using a versioned directory "
        "for Python modules.")
      set(_svtk_python_version_suffix)
    endif ()
    set(destination "${CMAKE_INSTALL_LIBDIR}/python${_svtk_python_version_suffix}/site-packages")
  endif ()

  set("${var}" "${destination}" PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module-impl
@brief Generate sources for using a module's classes from Python

This function generates the wrapped sources for a module. It places the list of
generated source files and classes in variables named in the second and third
arguments, respectively.

~~~
_svtk_module_wrap_python_sources(<module> <sources> <classes>)
~~~
#]==]
function (_svtk_module_wrap_python_sources module sources classes)
  _svtk_module_get_module_property("${module}"
    PROPERTY  "exclude_wrap"
    VARIABLE  _svtk_python_exclude_wrap)
  if (_svtk_python_exclude_wrap)
    return ()
  endif ()

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_library_name}Python")

  set(_svtk_python_args_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_library_name}Python/${_svtk_python_library_name}-python.$<CONFIGURATION>.args")

  set(_svtk_python_hierarchy_depends "${module}")
  _svtk_module_get_module_property("${module}"
    PROPERTY  "private_depends"
    VARIABLE  _svtk_python_private_depends)
  list(APPEND _svtk_python_hierarchy_depends ${_svtk_python_private_depends})
  
  set(_svtk_python_command_depends)
  foreach (_svtk_python_hierarchy_depend IN LISTS _svtk_python_hierarchy_depends)
    _svtk_module_get_module_property("${_svtk_python_hierarchy_depend}"
      PROPERTY  "hierarchy"
      VARIABLE  _svtk_python_hierarchy_file)
    if (_svtk_python_hierarchy_file)
      list(APPEND _svtk_python_hierarchy_files "${_svtk_python_hierarchy_file}")
      get_property(_svtk_python_is_imported
        TARGET    "${_svtk_python_hierarchy_depend}"
        PROPERTY  "IMPORTED")
      if (_svtk_python_is_imported OR CMAKE_GENERATOR MATCHES "Ninja")
        list(APPEND _svtk_python_command_depends "${_svtk_python_hierarchy_file}")
      else ()
        _svtk_module_get_module_property("${_svtk_python_hierarchy_depend}"
          PROPERTY  "library_name"
          VARIABLE  _svtk_python_hierarchy_library_name)
        if (TARGET "${_svtk_python_hierarchy_library_name}-hierarchy")
          list(APPEND _svtk_python_command_depends "${_svtk_python_hierarchy_library_name}-hierarchy")
        else ()
          message(FATAL_ERROR
            "The ${_svtk_python_hierarchy_depend} hierarchy file is attached to a non-imported target "
            "and a hierarchy target (${_svtk_python_hierarchy_library_name}-hierarchy) is "
            "missing.")
        endif ()
      endif ()
    endif ()
  endforeach ()

  set(_svtk_python_genex_compile_definitions
    "$<TARGET_PROPERTY:${_svtk_python_target_name},COMPILE_DEFINITIONS>")
  set(_svtk_python_genex_include_directories
    "$<TARGET_PROPERTY:${_svtk_python_target_name},INCLUDE_DIRECTORIES>")
  file(GENERATE
    OUTPUT  "${_svtk_python_args_file}"
    CONTENT "$<$<BOOL:${_svtk_python_genex_compile_definitions}>:\n-D\'$<JOIN:${_svtk_python_genex_compile_definitions},\'\n-D\'>\'>\n
$<$<BOOL:${_svtk_python_genex_include_directories}>:\n-I\'$<JOIN:${_svtk_python_genex_include_directories},\'\n-I\'>\'>\n
$<$<BOOL:${_svtk_python_hierarchy_files}>:\n--types \'$<JOIN:${_svtk_python_hierarchy_files},\'\n--types \'>\'>\n")

  set(_svtk_python_sources)

  # Get the list of public headers from the module.
  _svtk_module_get_module_property("${module}"
    PROPERTY  "headers"
    VARIABLE  _svtk_python_headers)
  set(_svtk_python_classes)
  foreach (_svtk_python_header IN LISTS _svtk_python_headers)
    # Assume the class name matches the basename of the header. This is SVTK
    # convention.
    get_filename_component(_svtk_python_basename "${_svtk_python_header}" NAME_WE)
    list(APPEND _svtk_python_classes
      "${_svtk_python_basename}")

    set(_svtk_python_source_output
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_library_name}Python/${_svtk_python_basename}Python.cxx")
    list(APPEND _svtk_python_sources
      "${_svtk_python_source_output}")

    set(_svtk_python_wrap_target "SVTK::WrapPython")
    set(_svtk_python_macros_args)
    if (TARGET SVTKCompileTools::WrapPython)
      set(_svtk_python_wrap_target "SVTKCompileTools::WrapPython")
      if (TARGET SVTKCompileTools_macros)
        list(APPEND _svtk_python_command_depends
          "SVTKCompileTools_macros")
        list(APPEND _svtk_python_macros_args
          -undef
          -imacros "${_SVTKCompileTools_macros_file}")
      endif ()
    endif ()

    add_custom_command(
      OUTPUT  "${_svtk_python_source_output}"
      COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
              "$<TARGET_FILE:${_svtk_python_wrap_target}>"
              "@${_svtk_python_args_file}"
              -o "${_svtk_python_source_output}"
              "${_svtk_python_header}"
              ${_svtk_python_macros_args}
      IMPLICIT_DEPENDS
              CXX "${_svtk_python_header}"
      COMMENT "Generating Python wrapper sources for ${_svtk_python_basename}"
      DEPENDS
        "${_svtk_python_header}"
        "${_svtk_python_args_file}"
        "$<TARGET_FILE:${_svtk_python_wrap_target}>"
        ${_svtk_python_command_depends})
  endforeach ()

  set("${sources}"
    "${_svtk_python_sources}"
    PARENT_SCOPE)
  set("${classes}"
    "${_svtk_python_classes}"
    PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module-impl
@brief Generate a CPython library for a set of modules

A Python module library may consist of the Python wrappings of multiple
modules. This is useful for kit-based builds where the modules part of the same
kit belong to the same Python module as well.

~~~
_svtk_module_wrap_python_library(<name> <module>...)
~~~

The first argument is the name of the Python module. The remaining arguments
are modules to include in the Python module.

The remaining information it uses is assumed to be provided by the
@ref svtk_module_wrap_python function.
#]==]
function (_svtk_module_wrap_python_library name)
  set(_svtk_python_library_sources)
  set(_svtk_python_library_classes)
  foreach (_svtk_python_module IN LISTS ARGN)
    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "exclude_wrap"
      VARIABLE  _svtk_python_exclude_wrap)
    if (_svtk_python_exclude_wrap)
      continue ()
    endif ()
    _svtk_module_real_target(_svtk_python_target_name "${_svtk_python_module}")
    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_python_library_name)

    # Wrap the module independently of the other SVTK modules in the Python
    # module.
    _svtk_module_wrap_python_sources("${_svtk_python_module}" _svtk_python_sources _svtk_python_classes)
    list(APPEND _svtk_python_library_sources
      ${_svtk_python_sources})
    list(APPEND _svtk_python_library_classes
      ${_svtk_python_classes})

    # Make sure the module doesn't already have an associated Python package.
    svtk_module_get_property("${_svtk_python_module}"
      PROPERTY  "INTERFACE_svtk_module_python_package"
      VARIABLE  _svtk_python_current_python_package)
    if (DEFINED _svtk_python_current_python_package)
      message(FATAL_ERROR
        "It appears as though the ${_svtk_python_module} has already been "
        "wrapped in Python in the ${_svtk_python_current_python_package} "
        "package.")
    endif ()
    svtk_module_set_property("${_svtk_python_module}"
      PROPERTY  "INTERFACE_svtk_module_python_package"
      VALUE     "${_svtk_python_PYTHON_PACKAGE}")

    if (_svtk_python_INSTALL_HEADERS)
      _svtk_module_export_properties(
        BUILD_FILE    "${_svtk_python_properties_build_file}"
        INSTALL_FILE  "${_svtk_python_properties_install_file}"
        MODULE        "${_svtk_python_module}"
        PROPERTIES
          # Export the wrapping hints file.
          INTERFACE_svtk_module_python_package)
    endif ()
  endforeach ()

  # The foreach needs to be split so that dependencies are guaranteed to have
  # the INTERFACE_svtk_module_python_package property set.
  foreach (_svtk_python_module IN LISTS ARGN)
    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "exclude_wrap"
      VARIABLE  _svtk_python_exclude_wrap)
    if (_svtk_python_exclude_wrap)
      continue ()
    endif ()

    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_python_library_name)

    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "depends"
      VARIABLE  _svtk_python_module_depends)
    set(_svtk_python_module_load_depends)
    foreach (_svtk_python_module_depend IN LISTS _svtk_python_module_depends)
      _svtk_module_get_module_property("${_svtk_python_module_depend}"
        PROPERTY  "exclude_wrap"
        VARIABLE  _svtk_python_module_depend_exclude_wrap)
      if (_svtk_python_module_depend_exclude_wrap)
        continue ()
      endif ()

      _svtk_module_get_module_property("${_svtk_python_module_depend}"
        PROPERTY  "python_package"
        VARIABLE  _svtk_python_depend_module_package)
      _svtk_module_get_module_property("${_svtk_python_module_depend}"
        PROPERTY  "library_name"
        VARIABLE  _svtk_python_depend_library_name)

      # XXX(kits): This doesn't work for kits.
      list(APPEND _svtk_python_module_load_depends
        "${_svtk_python_depend_module_package}.${_svtk_python_depend_library_name}")
    endforeach ()

    if (_svtk_python_BUILD_STATIC)
      # If static, we use .py modules that grab the contents from the baked-in modules.
      set(_svtk_python_module_file
        "${CMAKE_BINARY_DIR}/${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}/${_svtk_python_library_name}.py")
      set(_svtk_python_module_contents
          "from ${_svtk_python_import_prefix}${_svtk_python_library_name} import *\n")

      file(GENERATE
        OUTPUT  "${_svtk_python_module_file}"
        CONTENT "${_svtk_python_module_contents}")

      # Set `python_modules` to provide the list of python files that go along with
      # this module
      _svtk_module_set_module_property("${_svtk_python_module}" APPEND
        PROPERTY  "python_modules"
        VALUE     "${_svtk_python_module_file}")
    endif ()
  endforeach ()

  if (NOT _svtk_python_library_sources)
    return ()
  endif ()

  set(_svtk_python_init_data_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}Python/${name}-init.data")

  file(GENERATE
    OUTPUT  "${_svtk_python_init_data_file}"
    CONTENT "${_svtk_python_library_name}\n$<JOIN:${_svtk_python_classes},\n>\nDEPENDS\n$<JOIN:${_svtk_python_module_load_depends},\n>\n")

  set(_svtk_python_init_output
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}Python/${name}Init.cxx")
  set(_svtk_python_init_impl_output
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}Python/${name}InitImpl.cxx")
  list(APPEND _svtk_python_library_sources
    "${_svtk_python_init_output}"
    "${_svtk_python_init_impl_output}")

  set(_svtk_python_wrap_target "SVTK::WrapPythonInit")
  if (TARGET SVTKCompileTools::WrapPythonInit)
    set(_svtk_python_wrap_target "SVTKCompileTools::WrapPythonInit")
  endif ()

  if(_svtk_python_BUILD_STATIC)
    set(additonal_options "${_svtk_python_import_prefix}")
  endif()
  add_custom_command(
    OUTPUT  "${_svtk_python_init_output}"
            "${_svtk_python_init_impl_output}"
    COMMAND "${_svtk_python_wrap_target}"
            "${_svtk_python_init_data_file}"
            "${_svtk_python_init_output}"
            "${_svtk_python_init_impl_output}"
            "${additonal_options}"
    COMMENT "Generating the Python module initialization sources for ${name}"
    DEPENDS
      "${_svtk_python_init_data_file}"
      "$<TARGET_FILE:${_svtk_python_wrap_target}>")

  if (_svtk_python_BUILD_STATIC)
    set(_svtk_python_module_header_file
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}/static_python/${name}.h")
    set(_svtk_python_module_header_content
"#ifndef ${name}_h
#define ${name}_h

#include <svtkPython.h>

#ifdef __cplusplus
extern \"C\" {
#endif
#if PY_VERSION_HEX < 0x03000000
extern void init${_svtk_python_library_name}();
#else
extern PyObject* PyInit_${_svtk_python_library_name}();
#endif
#ifdef __cplusplus
}
#endif

#endif
")

    file(GENERATE
      OUTPUT  "${_svtk_python_module_header_file}"
      CONTENT "${_svtk_python_module_header_content}")
    # XXX(cmake): Why is this necessary? One would expect that `file(GENERATE)`
    # would do this automatically.
    set_property(SOURCE "${_svtk_python_module_header_file}"
      PROPERTY
        GENERATED 1)

    add_library("${name}" STATIC
      ${_svtk_python_library_sources}
      "${_svtk_python_module_header_file}")
    target_include_directories("${name}"
      INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}/static_python>")
    target_link_libraries("${name}"
      PUBLIC
        SVTK::Python)
    set_property(TARGET "${name}"
      PROPERTY
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_python_STATIC_MODULE_DESTINATION}")
  else ()
    add_library("${name}" MODULE
      ${_svtk_python_library_sources})
    if (WIN32 AND NOT CYGWIN)
      # XXX(python-debug): This is disabled out because there's no reliable way
      # to tell whether we're using a debug build of Python or not. Since using
      # a debug Python build is so rare, just assume we're always using a
      # non-debug build of Python itself.
      #
      # The proper fix is to dig around and ask the backing `PythonN::Python`
      # target used by `SVTK::Python` for its properties to find out, per
      # configuration, whether it is a debug build. If it is, add the postfix
      # (regardless of SVTK's build type). Otherwise, no postfix.
      if (FALSE)
        set_property(TARGET "${name}"
          APPEND_STRING
          PROPERTY
            DEBUG_POSTFIX "_d")
      endif ()
      set_property(TARGET "${name}"
        PROPERTY
          SUFFIX ".pyd")
    endif ()
    set_property(TARGET "${name}"
      PROPERTY
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}")
    get_property(_svtk_python_is_multi_config GLOBAL
      PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (_svtk_python_is_multi_config)
      # XXX(MultiNinja): This isn't going to work in general since MultiNinja
      # will error about overlapping output paths.
      foreach (_svtk_python_config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_svtk_python_config}" _svtk_python_config_upper)
        set_property(TARGET "${name}"
          PROPERTY
            "LIBRARY_OUTPUT_DIRECTORY_${_svtk_python_config_upper}" "${CMAKE_BINARY_DIR}/${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}")
      endforeach ()
    endif ()
    set_target_properties("${name}"
      PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${_svtk_python_library_name}"
        ARCHIVE_OUTPUT_NAME "${name}")
  endif ()

  svtk_module_autoinit(
    MODULES ${ARGN}
    TARGETS "${name}")

  # The wrapper code will expand PYTHON_PACKAGE as needed
  target_compile_definitions("${name}"
    PRIVATE
      "-DPYTHON_PACKAGE=\"${_svtk_python_PYTHON_PACKAGE}\"")

  target_link_libraries("${name}"
    PRIVATE
      ${ARGN}
      SVTK::WrappingPythonCore
      SVTK::Python)

  set(_svtk_python_export)
  if (_svtk_python_INSTALL_EXPORT)
    set(_svtk_python_export
      EXPORT "${_svtk_python_INSTALL_EXPORT}")
  endif ()

  install(
    TARGETS             "${name}"
    ${_svtk_python_export}
    COMPONENT           "${_svtk_python_COMPONENT}"
    RUNTIME DESTINATION "${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}"
    LIBRARY DESTINATION "${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}"
    ARCHIVE DESTINATION "${_svtk_python_STATIC_MODULE_DESTINATION}")
endfunction ()

#[==[
@ingroup module-wrapping-python
@brief Wrap a set of modules for use in Python

~~~
svtk_module_wrap_python(
  MODULES <module>...
  [TARGET <target>]
  [WRAPPED_MODULES <varname>]

  [BUILD_STATIC <ON|OFF>]
  [INSTALL_HEADERS <ON|OFF>]

  [DEPENDS <target>...]

  [MODULE_DESTINATION <destination>]
  [STATIC_MODULE_DESTINATION <destination>]
  [CMAKE_DESTINATION <destination>]
  [LIBRARY_DESTINATION <destination>]

  [PYTHON_PACKAGE <package>]
  [SOABI <soabi>]

  [INSTALL_EXPORT <export>]
  [COMPONENT <component>])
~~~

  * `MODULES`: (Required) The list of modules to wrap.
  * `TARGET`: (Recommended) The target to create which represents all wrapped
    Python modules. This is mostly useful when supporting static Python modules
    in order to add the generated modules to the built-in table.
  * `WRAPPED_MODULES`: (Recommended) Not all modules are wrappable. This
    variable will be set to contain the list of modules which were wrapped.
    These modules will have a `INTERFACE_svtk_module_python_package` property
    set on them which is the name that should be given to `import` statements
    in Python code.
  * `BUILD_STATIC`: Defaults to `${BUILD_SHARED_LIBS}`. Note that shared
    modules with a static build is not completely supported. For static Python
    module builds, a header named `<TARGET>.h` will be available with a
    function `void <TARGET>_load()` which will add all Python modules created
    by this call to the imported module table. For shared Python module builds,
    the same function is provided, but it is a no-op.
  * `INSTALL_HEADERS` (Defaults to `ON`): If unset, CMake properties will not
    be installed.
  * `DEPENDS`: This is list of other Python modules targets i.e. targets
    generated from previous calls to `svtk_module_wrap_python` that this new
    target depends on. This is used when `BUILD_STATIC` is true to ensure that
    the `void <TARGET>_load()` is correctly called for each of the dependencies.
  * `MODULE_DESTINATION`: Modules will be placed in this location in the
    build tree. The install tree should remove `$<CONFIGURATION>` bits, but it
    currently does not. See `svtk_module_python_default_destination` for the
    default value.
  * `STATIC_MODULE_DESTINATION`: Defaults to `${CMAKE_INSTALL_LIBDIR}`. This
    default may change in the future since the best location for these files is
    not yet known. Static libraries containing Python code will be installed to
    the install tree under this path.
  * `CMAKE_DESTINATION`: (Required if `INSTALL_HEADERS` is `ON`) Where to
    install Python-related module property CMake files.
  * `LIBRARY_DESTINATION` (Recommended): If provided, dynamic loader
    information will be added to modules for loading dependent libraries.
  * `PYTHON_PACKAGE`: (Recommended) All generated modules will be added to this
    Python package. The format is in Python syntax (e.g.,
    `package.subpackage`).
  * `SOABI`: (Required for wheel support): If given, generate libraries with
    the SOABI tag in the module filename.
  * `INSTALL_EXPORT`: If provided, static installs will add the installed
    libraries to the provided export set.
  * `COMPONENT`: Defaults to `python`. All install rules created by this
    function will use this installation component.
#]==]
function (svtk_module_wrap_python)
  cmake_parse_arguments(_svtk_python
    ""
    "MODULE_DESTINATION;STATIC_MODULE_DESTINATION;LIBRARY_DESTINATION;PYTHON_PACKAGE;BUILD_STATIC;INSTALL_HEADERS;INSTALL_EXPORT;TARGET;COMPONENT;WRAPPED_MODULES;CMAKE_DESTINATION;DEPENDS;SOABI"
    "MODULES"
    ${ARGN})

  if (_svtk_python_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_wrap_python: "
      "${_svtk_python_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT _svtk_python_MODULES)
    message(WARNING
      "No modules were requested for Python wrapping.")
    return ()
  endif ()

  _svtk_module_split_module_name("${_svtk_python_TARGET}" _svtk_python)

  set(_svtk_python_depends)
  foreach (_svtk_python_depend IN LISTS _svtk_python_DEPENDS)
    _svtk_module_split_module_name("${_svtk_python_depend}" _svtk_python_depends)
    list(APPEND _svtk_python_depends
      "${_svtk_python_depends_TARGET_NAME}")
  endforeach ()

  if (NOT DEFINED _svtk_python_MODULE_DESTINATION)
    svtk_module_python_default_destination(_svtk_python_MODULE_DESTINATION)
  endif ()

  if (NOT DEFINED _svtk_python_INSTALL_HEADERS)
    set(_svtk_python_INSTALL_HEADERS ON)
  endif ()

  if (_svtk_python_SOABI)
    get_property(_svtk_python_is_multi_config GLOBAL
      PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (_svtk_python_is_multi_config)
      foreach (_svtk_python_config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_svtk_python_config}" _svtk_python_upper_config)
        set("CMAKE_${_svtk_python_upper_config}_POSTFIX"
          ".${_svtk_python_SOABI}")
      endforeach ()
    else ()
      string(TOUPPER "${CMAKE_BUILD_TYPE}" _svtk_python_upper_config)
      set("CMAKE_${_svtk_python_upper_config}_POSTFIX"
        ".${_svtk_python_SOABI}")
    endif ()
  endif ()

  if (_svtk_python_INSTALL_HEADERS AND NOT DEFINED _svtk_python_CMAKE_DESTINATION)
    message(FATAL_ERROR
      "No CMAKE_DESTINATION set, but headers from the Python wrapping were "
      "requested for install and the CMake files are required to work with "
      "them.")
  endif ()

  if (NOT DEFINED _svtk_python_BUILD_STATIC)
    if (BUILD_SHARED_LIBS)
      set(_svtk_python_BUILD_STATIC OFF)
    else ()
      set(_svtk_python_BUILD_STATIC ON)
    endif ()
  else ()
    if (NOT _svtk_python_BUILD_STATIC AND NOT BUILD_SHARED_LIBS)
      message(WARNING
        "Building shared Python modules against static SVTK modules only "
        "supports consuming the SVTK modules via their Python interfaces due "
        "to the lack of support for an SDK to use the same static libraries.")
    endif ()
  endif ()

  if (NOT DEFINED _svtk_python_STATIC_MODULE_DESTINATION)
    # TODO: Is this correct?
    set(_svtk_python_STATIC_MODULE_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif ()

  if (NOT DEFINED _svtk_python_COMPONENT)
    set(_svtk_python_COMPONENT "python")
  endif ()

  if (NOT _svtk_python_PYTHON_PACKAGE)
    message(FATAL_ERROR
      "No `PYTHON_PACKAGE` was given; Python modules must be placed into a "
      "package.")
  endif ()
  string(REPLACE "." "/" _svtk_python_package_path "${_svtk_python_PYTHON_PACKAGE}")

  if(_svtk_python_BUILD_STATIC)
    # When doing static builds we want the statically initialized built-ins to be
    # used. It is unclear in the Python-C API how to construct `namespace.module`
    # so instead at the C++ level we import "namespace_module" during startup
    # and than the python modules moving those imports into the correct python
    # module.
    string(REPLACE "." "_" _svtk_python_import_prefix "${_svtk_python_PYTHON_PACKAGE}_")
  else()
    # We are building dynamic libraries therefore the prefix is simply '.'
    set(_svtk_python_import_prefix ".")
  endif()

  _svtk_module_check_destinations(_svtk_python_
    MODULE_DESTINATION
    STATIC_MODULE_DESTINATION
    CMAKE_DESTINATION
    LIBRARY_DESTINATION)

  if (_svtk_python_INSTALL_HEADERS)
    set(_svtk_python_properties_filename "${_svtk_python_PYTHON_PACKAGE}-svtk-python-module-properties.cmake")
    set(_svtk_python_properties_install_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_TARGET_NAME}/${_svtk_python_properties_filename}.install")
    set(_svtk_python_properties_build_file "${CMAKE_BINARY_DIR}/${_svtk_python_CMAKE_DESTINATION}/${_svtk_python_properties_filename}")

    file(WRITE "${_svtk_python_properties_build_file}")
    file(WRITE "${_svtk_python_properties_install_file}")
  endif ()

  if (DEFINED _svtk_python_LIBRARY_DESTINATION)
    # Set up rpaths
    set(CMAKE_BUILD_RPATH_USE_ORIGIN 1)
    if (UNIX)
      file(RELATIVE_PATH _svtk_python_relpath
        "/prefix/${_svtk_python_MODULE_DESTINATION}/${_svtk_python_package_path}"
        "/prefix/${_svtk_python_LIBRARY_DESTINATION}")

      if (APPLE)
        set(_svtk_python_origin_stem "@loader_path")
      else ()
        set(_svtk_python_origin_stem "$ORIGIN")
      endif()

      list(APPEND CMAKE_INSTALL_RPATH
        "${_svtk_python_origin_stem}/${_svtk_python_relpath}")
    endif ()
  endif ()

  set(_svtk_python_sorted_modules ${_svtk_python_MODULES})
  foreach (_svtk_python_module IN LISTS _svtk_python_MODULES)
    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "depends"
      VARIABLE  "_svtk_python_${_svtk_python_module}_depends")
  endforeach ()
  svtk_topological_sort(_svtk_python_sorted_modules "_svtk_python_" "_depends")

  set(_svtk_python_sorted_modules_filtered)
  foreach (_svtk_python_module IN LISTS _svtk_python_sorted_modules)
    if (_svtk_python_module IN_LIST _svtk_python_MODULES)
      list(APPEND _svtk_python_sorted_modules_filtered
        "${_svtk_python_module}")
    endif ()
  endforeach ()

  set(_svtk_python_all_modules)
  set(_svtk_python_all_wrapped_modules)
  foreach (_svtk_python_module IN LISTS _svtk_python_sorted_modules_filtered)
    _svtk_module_get_module_property("${_svtk_python_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_python_library_name)
    _svtk_module_wrap_python_library("${_svtk_python_library_name}Python" "${_svtk_python_module}")

    if (TARGET "${_svtk_python_library_name}Python")
      list(APPEND _svtk_python_all_modules
        "${_svtk_python_library_name}Python")
      list(APPEND _svtk_python_all_wrapped_modules
        "${_svtk_python_module}")
    endif ()
  endforeach ()

  if (NOT _svtk_python_all_modules)
    message(FATAL_ERROR
      "No modules given could be wrapped.")
  endif ()

  if (_svtk_python_INSTALL_HEADERS)
    install(
      FILES       "${_svtk_python_properties_install_file}"
      DESTINATION "${_svtk_python_CMAKE_DESTINATION}"
      RENAME      "${_svtk_python_properties_filename}"
      COMPONENT   "development")
  endif ()

  if (DEFINED _svtk_python_WRAPPED_MODULES)
    set("${_svtk_python_WRAPPED_MODULES}"
      "${_svtk_python_all_wrapped_modules}"
      PARENT_SCOPE)
  endif ()

  if (_svtk_python_TARGET)
    add_library("${_svtk_python_TARGET_NAME}" INTERFACE)
    target_include_directories("${_svtk_python_TARGET_NAME}"
      INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_TARGET_NAME}/static_python>")
    target_link_libraries("${_svtk_python_TARGET_NAME}"
      INTERFACE
        ${_svtk_python_DEPENDS})
    if (NOT _svtk_python_TARGET STREQUAL _svtk_python_TARGET_NAME)
      add_library("${_svtk_python_TARGET}" ALIAS
        "${_svtk_python_TARGET_NAME}")
    endif ()

    if (_svtk_python_INSTALL_EXPORT)
      install(
        TARGETS   "${_svtk_python_TARGET_NAME}"
        EXPORT    "${_svtk_python_INSTALL_EXPORT}"
        COMPONENT "development")
    endif ()

    set(_svtk_python_all_modules_include_file
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_TARGET_NAME}/static_python/${_svtk_python_TARGET_NAME}.h")
    set(_svtk_python_all_modules_include_content
      "#ifndef ${_svtk_python_TARGET_NAME}_h\n#define ${_svtk_python_TARGET_NAME}_h\n")

    if (_svtk_python_BUILD_STATIC)
      foreach (_svtk_python_module IN LISTS _svtk_python_all_modules)
        string(APPEND _svtk_python_all_modules_include_content
          "#include \"${_svtk_python_module}.h\"\n")
      endforeach ()
    endif ()

    foreach (_svtk_python_depend IN LISTS _svtk_python_depends)
      string(APPEND _svtk_python_all_modules_include_content
        "#include \"${_svtk_python_depend}.h\"\n")
    endforeach ()

    string(APPEND _svtk_python_all_modules_include_content
"#if PY_VERSION_HEX < 0x03000000
#define PY_APPEND_INIT(module) PyImport_AppendInittab(\"${_svtk_python_import_prefix}\" #module, init ## module)
#define PY_IMPORT(module) init ## module();
#else
#define PY_APPEND_INIT(module) PyImport_AppendInittab(\"${_svtk_python_import_prefix}\" #module, PyInit_ ## module)
#define PY_IMPORT(module) { \\
    PyObject* var_ ## module = PyInit_ ## module(); \\
    PyDict_SetItemString(PyImport_GetModuleDict(), \"${_svtk_python_import_prefix}\" #module,var_ ## module); \\
    Py_DECREF(var_ ## module); }
#endif

#define PY_APPEND_INIT_OR_IMPORT(module, do_import) \\
  if (do_import) { PY_IMPORT(module); } else { PY_APPEND_INIT(module); }

static void ${_svtk_python_TARGET_NAME}_load() {\n")

    foreach (_svtk_python_depend IN LISTS _svtk_python_depends)
      string(APPEND _svtk_python_all_modules_include_content
        "  ${_svtk_python_depend}_load();\n")
    endforeach ()

    if (_svtk_python_BUILD_STATIC)
      string(APPEND _svtk_python_all_modules_include_content
        "  int do_import = Py_IsInitialized();\n")
      foreach (_svtk_python_module IN LISTS _svtk_python_sorted_modules_filtered)
        _svtk_module_get_module_property("${_svtk_python_module}"
          PROPERTY  "library_name"
          VARIABLE  _svtk_python_library_name)
        if (TARGET "${_svtk_python_library_name}Python")
          string(APPEND _svtk_python_all_modules_include_content
            "  PY_APPEND_INIT_OR_IMPORT(${_svtk_python_library_name}, do_import);\n")
        endif ()
      endforeach ()
    endif ()

    string(APPEND _svtk_python_all_modules_include_content
      "}\n#undef PY_APPEND_INIT\n#undef PY_IMPORT\n#undef PY_APPEND_INIT_OR_IMPORT\n#endif\n")

    # TODO: Install this header.
    file(GENERATE
      OUTPUT  "${_svtk_python_all_modules_include_file}"
      CONTENT "${_svtk_python_all_modules_include_content}")

    if (_svtk_python_BUILD_STATIC)
      # TODO: Install these targets.
      target_link_libraries("${_svtk_python_TARGET_NAME}"
        INTERFACE
          ${_svtk_python_all_modules})
    endif ()

    if (_svtk_python_BUILD_STATIC)
      # Next, we generate a Python module that can be imported to import any
      # static artifacts e.g. all wrapping Python modules in static builds,
      # (eventually, frozen modules etc.)
      string(REPLACE "." "_" _svtk_python_static_importer_name "_${_svtk_python_PYTHON_PACKAGE}_static")
      set(_svtk_python_static_importer_file
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_python_TARGET_NAME}/static_python/${_svtk_python_static_importer_name}.c")
      set(_svtk_python_static_importer_content "// generated file, do not edit!
#include <svtkPython.h>
#include \"${_svtk_python_TARGET_NAME}.h\"

  static PyMethodDef Py${_svtk_python_static_importer_name}_Methods[] = {
  {NULL, NULL, 0, NULL}};
#if PY_VERSION_HEX >= 0x03000000
  static PyModuleDef ${_svtk_python_static_importer_name}Module = {
    PyModuleDef_HEAD_INIT,
    \"${_svtk_python_static_importer_name}\", // m_name
    \"module to import static components for ${_svtk_python_TARGET_NAME}\", // m_doc
    0, // m_size
    Py${_svtk_python_static_importer_name}_Methods, // m_methods
    NULL, // m_reload
    NULL, // m_traverse
    NULL, // m_clear
    NULL  // m_free
  };
#endif

#if PY_VERSION_HEX >= 0x03000000
  PyMODINIT_FUNC PyInit_${_svtk_python_static_importer_name}(void)
#else
  PyMODINIT_FUNC init${_svtk_python_static_importer_name}(void)
#endif
  {
    // since this gets called after `Py_Initialize`, this will import the static
    // modules and not just update the init table.
    ${_svtk_python_TARGET_NAME}_load();
#if PY_VERSION_HEX >= 0x03000000
    return PyModule_Create(&${_svtk_python_static_importer_name}Module);
#else
    Py_InitModule(\"${_svtk_python_static_importer_name}\", Py${_svtk_python_static_importer_name}_Methods);
#endif
  }\n")

      # TODO: Install this header.
      file(GENERATE
        OUTPUT  "${_svtk_python_static_importer_file}"
        CONTENT "${_svtk_python_static_importer_content}")

      add_library("${_svtk_python_static_importer_name}" MODULE
        ${_svtk_python_static_importer_file})
      if (WIN32 AND NOT CYGWIN)
        set_property(TARGET "${_svtk_python_static_importer_name}"
          PROPERTY
            SUFFIX ".pyd")
      endif()
      set_property(TARGET "${_svtk_python_static_importer_name}"
        PROPERTY
          LIBRARY_OUTPUT_DIRECTORY "${_svtk_python_MODULE_DESTINATION}")
      get_property(_svtk_python_is_multi_config GLOBAL
        PROPERTY GENERATOR_IS_MULTI_CONFIG)
      if (_svtk_python_is_multi_config)
        # XXX(MultiNinja): This isn't going to work in general since MultiNinja
        # will error about overlapping output paths.
        foreach (_svtk_python_config IN LISTS CMAKE_CONFIGURATION_TYPES)
          string(TOUPPER "${_svtk_python_config}" _svtk_python_config_upper)
          set_property(TARGET "${_svtk_python_static_importer_name}"
            PROPERTY
              "LIBRARY_OUTPUT_DIRECTORY_${_svtk_python_config_upper}" "${CMAKE_BINARY_DIR}/${_svtk_python_MODULE_DESTINATION}")
        endforeach ()
      endif ()
      set_property(TARGET "${_svtk_python_static_importer_name}"
        PROPERTY
          PREFIX "")
      target_link_libraries("${_svtk_python_static_importer_name}"
        PRIVATE
          ${_svtk_python_TARGET_NAME}
          SVTK::WrappingPythonCore
          SVTK::Python)
      install(
        TARGETS             "${_svtk_python_static_importer_name}"
        COMPONENT           "${_svtk_python_COMPONENT}"
        RUNTIME DESTINATION "${_svtk_python_MODULE_DESTINATION}"
        LIBRARY DESTINATION "${_svtk_python_MODULE_DESTINATION}"
        ARCHIVE DESTINATION "${_svtk_python_STATIC_MODULE_DESTINATION}")
    endif () # if (_svtk_python_BUILD_STATIC)
  endif ()
endfunction ()

#[==[
@ingroup module-wrapping-python
@brief Install Python packages with a module

Some modules may have associated Python code. This function should be used to
install them.

~~~
svtk_module_add_python_package(<module>
  PACKAGE <package>
  FILES <files>...
  [MODULE_DESTINATION <destination>]
  [COMPONENT <component>])
~~~

The `<module>` argument must match the associated SVTK module that the package
is with. Each package is independent and should be installed separately. That
is, `package` and `package.subpackage` should each get their own call to this
function.

  * `PACKAGE`: (Required) The package installed by this call. Currently,
    subpackages must have their own call to this function.
  * `FILES`: (Required) File paths should be relative to the source directory
    of the calling `CMakeLists.txt`. Upward paths are not supported (nor are
    checked for). Absolute paths are assumed to be in the build tree and their
    relative path is computed relative to the current binary directory.
  * `MODULE_DESTINATION`: Modules will be placed in this location in the
    build tree. The install tree should remove `$<CONFIGURATION>` bits, but it
    currently does not. See `svtk_module_python_default_destination` for the
    default value.
  * `COMPONENT`: Defaults to `python`. All install rules created by this
    function will use this installation component.

A `<module>-<package>` target is created which ensures that all Python modules
have been copied to the correct location in the build tree.

@todo Support a tree of modules with a single call.

@todo Support freezing the Python package. This should create a header and the
associated target should provide an interface for including this header. The
target should then be exported and the header installed properly.
#]==]
function (svtk_module_add_python_package name)
  if (NOT name STREQUAL _svtk_build_module)
    message(FATAL_ERROR
      "Python modules must match their module names.")
  endif ()

  cmake_parse_arguments(_svtk_add_python_package
    ""
    "PACKAGE;MODULE_DESTINATION;COMPONENT"
    "FILES"
    ${ARGN})

  if (_svtk_add_python_package_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_add_python_package: "
      "${_svtk_add_python_package_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT _svtk_add_python_package_PACKAGE)
    message(FATAL_ERROR
      "The `PACKAGE` argument is required.")
  endif ()
  string(REPLACE "." "/" _svtk_add_python_package_path "${_svtk_add_python_package_PACKAGE}")

  if (NOT _svtk_add_python_package_FILES)
    message(FATAL_ERROR
      "The `FILES` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_add_python_package_MODULE_DESTINATION)
    svtk_module_python_default_destination(_svtk_add_python_package_MODULE_DESTINATION)
  endif ()

  if (NOT DEFINED _svtk_add_python_package_COMPONENT)
    set(_svtk_add_python_package_COMPONENT "python")
  endif ()

  set(_svtk_add_python_package_file_outputs)
  foreach (_svtk_add_python_package_file IN LISTS _svtk_add_python_package_FILES)
    if (IS_ABSOLUTE "${_svtk_add_python_package_file}")
      file(RELATIVE_PATH _svtk_add_python_package_name
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${_svtk_add_python_package_name}")
    else ()
      set(_svtk_add_python_package_name
        "${_svtk_add_python_package_file}")
      set(_svtk_add_python_package_file
        "${CMAKE_CURRENT_SOURCE_DIR}/${_svtk_add_python_package_file}")
    endif ()

    set(_svtk_add_python_package_file_output
      "${CMAKE_BINARY_DIR}/${_svtk_add_python_package_MODULE_DESTINATION}/${_svtk_add_python_package_name}")
    add_custom_command(
      OUTPUT  "${_svtk_add_python_package_file_output}"
      DEPENDS "${_svtk_add_python_package_file}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${_svtk_add_python_package_file}"
              "${_svtk_add_python_package_file_output}"
      COMMENT "Copying ${_svtk_add_python_package_name} to the binary directory")
    list(APPEND _svtk_add_python_package_file_outputs
      "${_svtk_add_python_package_file_output}")
    # XXX
    if (BUILD_SHARED_LIBS)
      install(
        FILES       "${_svtk_add_python_package_name}"
        DESTINATION "${_svtk_add_python_package_MODULE_DESTINATION}/${_svtk_add_python_package_path}"
        COMPONENT   "${_svtk_add_python_package_COMPONENT}")
    endif()
  endforeach ()

  get_property(_svtk_add_python_package_module GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
  add_custom_target("${_svtk_add_python_package_module}-${_svtk_add_python_package_PACKAGE}" ALL
    DEPENDS
      ${_svtk_add_python_package_file_outputs})

  # Set `python_modules` to provide the list of python files that go along with
  # this module
  set_property(TARGET "${_svtk_add_python_package_module}-${_svtk_add_python_package_PACKAGE}"
    PROPERTY
      "python_modules" "${_svtk_add_python_package_file_outputs}")
endfunction ()

#[==[
@ingroup module-wrapping-python
@brief Use a Python package as a module

If a module is a Python package, this function should be used instead of
@ref svtk_module_add_module.

~~~
svtk_module_add_python_module(<name>
  PACKAGES <packages>...)
~~~

  * `PACKAGES`: (Required) The list of packages installed by this module.
    These must have been created by the @ref svtk_module_add_python_package
    function.
#]==]
function (svtk_module_add_python_module name)
  if (NOT name STREQUAL _svtk_build_module)
    message(FATAL_ERROR
      "Python modules must match their module names.")
  endif ()

  cmake_parse_arguments(_svtk_add_python_module
    ""
    ""
    "PACKAGES"
    ${ARGN})

  if (_svtk_add_python_module_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_add_python_module: "
      "${_svtk_add_python_module_UNPARSED_ARGUMENTS}")
  endif ()

  get_property(_svtk_add_python_module_depends GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_depends")
  get_property(_svtk_add_python_module_target_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
  add_library("${_svtk_add_python_module_target_name}" INTERFACE)
  target_link_libraries("${_svtk_add_python_module_target_name}"
    INTERFACE
      ${_svtk_add_python_module_depends})
  if (NOT _svtk_build_module STREQUAL _svtk_add_python_module_target_name)
    add_library("${_svtk_build_module}" ALIAS
      "${_svtk_add_python_module_target_name}")
  endif ()
  foreach (_svtk_add_python_module_package IN LISTS _svtk_add_python_module_PACKAGES)
    add_dependencies("${_svtk_add_python_module_target_name}"
      "${_svtk_build_module}-${_svtk_add_python_module_package}")

    # get the list of python files and add them on the module.
    get_property(_svtk_module_python_modules
      TARGET "${_svtk_add_python_module_target_name}-${_svtk_add_python_module_package}"
      PROPERTY "python_modules")
    _svtk_module_set_module_property("${_svtk_build_module}" APPEND
      PROPERTY  "python_modules"
      VALUE     "${_svtk_module_python_modules}")
  endforeach ()

  _svtk_module_apply_properties("${_svtk_add_python_module_target_name}")
  _svtk_module_install("${_svtk_add_python_module_target_name}")
endfunction ()

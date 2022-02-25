#[==[
@defgroup module-wrapping-java Module Java CMake APIs
#]==]

#[==[
@file svtkModuleWrapJava.cmake
@brief APIs for wrapping modules for Java
#]==]

#[==[
@ingroup module-impl
@brief Generate sources for using a module's classes from Java

This function generates the wrapped sources for a module. It places the list of
generated source files and Java source files in variables named in the second
and third arguments, respectively.

~~~
_svtk_module_wrap_java_sources(<module> <sources> <classes>)
~~~
#]==]
function (_svtk_module_wrap_java_sources module sources java_sources)
  _svtk_module_get_module_property("${module}"
    PROPERTY  "exclude_wrap"
    VARIABLE  _svtk_java_exclude_wrap)
  if (_svtk_java_exclude_wrap)
    return ()
  endif ()

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_java_library_name}Java")

  set(_svtk_java_args_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_java_library_name}Java/${_svtk_java_library_name}-java.$<CONFIGURATION>.args")
  set(_svtk_java_init_data_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_java_library_name}Java/${_svtk_java_library_name}-java-init.data")

  set(_svtk_java_hierarchy_depends "${module}")
  _svtk_module_get_module_property("${module}"
    PROPERTY  "private_depends"
    VARIABLE  _svtk_java_private_depends)
  list(APPEND _svtk_java_hierarchy_depends ${_svtk_java_private_depends})

  set(_svtk_java_command_depends)
  foreach (_svtk_java_hierarchy_depend IN LISTS _svtk_java_hierarchy_depends)
    _svtk_module_get_module_property("${_svtk_java_hierarchy_depend}"
      PROPERTY  "hierarchy"
      VARIABLE  _svtk_java_hierarchy_file)
    if (_svtk_java_hierarchy_file)
      list(APPEND _svtk_java_hierarchy_files "${_svtk_java_hierarchy_file}")
      get_property(_svtk_java_is_imported
        TARGET    "${_svtk_java_hierarchy_depend}"
        PROPERTY  "IMPORTED")
      if (_svtk_java_is_imported OR CMAKE_GENERATOR MATCHES "Ninja")
        list(APPEND _svtk_java_command_depends "${_svtk_java_hierarchy_file}")
      else ()
        _svtk_module_get_module_property("${_svtk_java_hierarchy_depend}"
          PROPERTY  "library_name"
          VARIABLE  _svtk_java_hierarchy_library_name)
        if (TARGET "${_svtk_java_hierarchy_library_name}-hierarchy")
          list(APPEND _svtk_java_command_depends "${_svtk_java_hierarchy_library_name}-hierarchy")
        else ()
          message(FATAL_ERROR
            "The ${_svtk_java_hierarchy_depend} hierarchy file is attached to a non-imported target "
            "and a hierarchy target (${_svtk_java_hierarchy_library_name}-hierarchy) is "
            "missing.")
        endif ()
      endif ()
    endif ()
  endforeach ()

  set(_svtk_java_genex_compile_definitions
    "$<TARGET_PROPERTY:${_svtk_java_target_name},COMPILE_DEFINITIONS>")
  set(_svtk_java_genex_include_directories
    "$<TARGET_PROPERTY:${_svtk_java_target_name},INCLUDE_DIRECTORIES>")
  file(GENERATE
    OUTPUT  "${_svtk_java_args_file}"
    CONTENT "$<$<BOOL:${_svtk_java_genex_compile_definitions}>:\n-D\'$<JOIN:${_svtk_java_genex_compile_definitions},\'\n-D\'>\'>\n
$<$<BOOL:${_svtk_java_genex_include_directories}>:\n-I\'$<JOIN:${_svtk_java_genex_include_directories},\'\n-I\'>\'>\n
$<$<BOOL:${_svtk_java_hierarchy_files}>:\n--types \'$<JOIN:${_svtk_java_hierarchy_files},\'\n--types \'>\'>\n")

  set(_svtk_java_sources)
  set(_svtk_java_java_sources)

  _svtk_module_get_module_property("${module}"
    PROPERTY  "headers"
    VARIABLE  _svtk_java_headers)
  set(_svtk_java_classes)
  foreach (_svtk_java_header IN LISTS _svtk_java_headers)
    get_filename_component(_svtk_java_basename "${_svtk_java_header}" NAME_WE)
    list(APPEND _svtk_java_classes
      "${_svtk_java_basename}")

    # The svtkWrapJava tool has special logic for the `svtkRenderWindow` class.
    # This extra logic requires its wrappers to be compiled as ObjC++ code
    # instead.
    set(_svtk_java_ext "cxx")
    if (APPLE AND _svtk_java_basename STREQUAL "svtkRenderWindow")
      set(_svtk_java_ext "mm")
    endif ()

    set(_svtk_java_source_output
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_java_library_name}Java/${_svtk_java_basename}Java.${_svtk_java_ext}")
    list(APPEND _svtk_java_sources
      "${_svtk_java_source_output}")

    set(_svtk_java_wrap_target "SVTK::WrapJava")
    set(_svtk_java_macros_args)
    if (TARGET SVTKCompileTools::WrapJava)
      set(_svtk_java_wrap_target "SVTKCompileTools::WrapJava")
      if (TARGET SVTKCompileTools_macros)
        list(APPEND _svtk_java_command_depends
          "SVTKCompileTools_macros")
        list(APPEND _svtk_java_macros_args
          -undef
          -imacros "${_SVTKCompileTools_macros_file}")
      endif ()
    endif ()

    set(_svtk_java_parse_target "SVTK::ParseJava")
    if (TARGET SVTKCompileTools::ParseJava)
      set(_svtk_java_parse_target "SVTKCompileTools::ParseJava")
    endif ()

    add_custom_command(
      OUTPUT  "${_svtk_java_source_output}"
      COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
              "$<TARGET_FILE:${_svtk_java_wrap_target}>"
              "@${_svtk_java_args_file}"
              -o "${_svtk_java_source_output}"
              "${_svtk_java_header}"
              ${_svtk_java_macros_args}
      IMPLICIT_DEPENDS
              CXX "${_svtk_java_header}"
      COMMENT "Generating Java wrapper sources for ${_svtk_java_basename}"
      DEPENDS
        "${_svtk_java_header}"
        "${_svtk_java_args_file}"
        "$<TARGET_FILE:${_svtk_java_wrap_target}>"
        ${_svtk_java_command_depends})

    set(_svtk_java_java_source_output
      "${_svtk_java_JAVA_OUTPUT}/${_svtk_java_basename}.java")
    list(APPEND _svtk_java_java_sources
      "${_svtk_java_java_source_output}")

    add_custom_command(
      OUTPUT  "${_svtk_java_java_source_output}"
      COMMAND "${_svtk_java_parse_target}"
              "@${_svtk_java_args_file}"
              -o "${_svtk_java_java_source_output}"
              "${_svtk_java_header}"
              ${_svtk_java_macros_args}
      IMPLICIT_DEPENDS
              CXX "${_svtk_java_header}"
      COMMENT "Generating Java sources for ${_svtk_java_basename}"
      DEPENDS
        "${_svtk_java_header}"
        "${_svtk_java_args_file}"
        "$<TARGET_FILE:${_svtk_java_parse_target}>"
        ${_svtk_java_command_depends})
  endforeach ()

  set("${sources}"
    "${_svtk_java_sources}"
    PARENT_SCOPE)

  set("${java_sources}"
    "${_svtk_java_java_sources}"
    PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module-impl
@brief Generate a JNI library for a set of modules

A single JNI library may consist of the Java wrappings of multiple modules.
This is useful for kit-based builds where the modules part of the same kit
belong to the same JNI library as well.

~~~
_svtk_module_wrap_java_library(<name> <module>...)
~~~

The first argument is the name of the JNI library. The remaining arguments are
modules to include in the JNI library.

The remaining information it uses is assumed to be provided by the
@ref svtk_module_wrap_java function.
#]==]
function (_svtk_module_wrap_java_library name)
  set(_svtk_java_library_sources)
  set(_svtk_java_library_java_sources)
  set(_svtk_java_library_link_depends)
  foreach (_svtk_java_module IN LISTS ARGN)
    _svtk_module_get_module_property("${_svtk_java_module}"
      PROPERTY  "exclude_wrap"
      VARIABLE  _svtk_java_exclude_wrap)
    if (_svtk_java_exclude_wrap)
      continue ()
    endif ()
    _svtk_module_real_target(_svtk_java_target_name "${_svtk_java_module}")
    _svtk_module_get_module_property("${_svtk_java_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_java_library_name)
    _svtk_module_wrap_java_sources("${_svtk_java_module}" _svtk_java_sources _svtk_java_java_sources)
    list(APPEND _svtk_java_library_sources
      ${_svtk_java_sources})
    list(APPEND _svtk_java_library_java_sources
      ${_svtk_java_java_sources})

    _svtk_module_get_module_property("${_svtk_java_module}"
      PROPERTY  "depends"
      VARIABLE  _svtk_java_module_depends)
    foreach (_svtk_java_module_depend IN LISTS _svtk_java_module_depends)
      _svtk_module_get_module_property("${_svtk_java_module_depend}"
        PROPERTY  "exclude_wrap"
        VARIABLE  _svtk_java_module_depend_exclude_wrap)
      if (_svtk_java_module_depend_exclude_wrap)
        continue ()
      endif ()

      _svtk_module_get_module_property("${_svtk_java_module_depend}"
        PROPERTY  "library_name"
        VARIABLE  _svtk_java_depend_library_name)

      # XXX(kits): This doesn't work for kits.
      list(APPEND _svtk_java_library_link_depends
        "${_svtk_java_depend_library_name}Java")
    endforeach ()
  endforeach ()

  if (NOT _svtk_java_library_sources)
    return ()
  endif ()

  if (_svtk_java_library_link_depends)
    list(REMOVE_DUPLICATES _svtk_java_library_link_depends)
  endif ()

  set(_svtk_java_target "${name}Java")

  # XXX(java): Should this be a `MODULE`? If not, we should probably export
  # these targets, but then we'll need logic akin to the `svtkModuleWrapPython`
  # logic for loading wrapped modules from other packages.
  add_library("${_svtk_java_target}" SHARED
    ${_svtk_java_library_sources})
  add_custom_target("${_svtk_java_target}-java-sources"
    DEPENDS
      ${_svtk_java_library_java_sources})
  add_dependencies("${_svtk_java_target}"
    "${_svtk_java_target}-java-sources")
  if (MINGW)
    set_property(TARGET "${_svtk_java_target}"
      PROPERTY
        PREFIX "")
  endif ()
  if (APPLE)
    set_property(TARGET "${_svtk_java_target}"
      PROPERTY
        SUFFIX ".jnilib")
  endif ()
  set_property(TARGET "${_svtk_java_target}"
    PROPERTY
      "_svtk_module_java_files" "${_svtk_java_library_java_sources}")

  if (_svtk_java_JNILIB_DESTINATION)
    install(
      TARGETS "${_svtk_java_target}"
      # Windows
      RUNTIME
        DESTINATION "${_svtk_java_JNILIB_DESTINATION}"
        COMPONENT   "${_svtk_java_JNILIB_COMPONENT}"
      # Other platforms
      LIBRARY
        DESTINATION "${_svtk_java_JNILIB_DESTINATION}"
        COMPONENT   "${_svtk_java_JNILIB_COMPONENT}")
  endif ()

  svtk_module_autoinit(
    MODULES ${ARGN}
    TARGETS "${_svtk_java_target}")

  target_link_libraries("${_svtk_java_target}"
    PRIVATE
      ${ARGN}
      # XXX(java): If we use modules, remove this.
      ${_svtk_java_library_link_depends}
      SVTK::Java)
endfunction ()

#[==[
@ingroup module-wrapping-java
@brief Wrap a set of modules for use in Java

~~~
svtk_module_wrap_java(
  MODULES <module>...
  [WRAPPED_MODULES <varname>]

  [JAVA_OUTPUT <destination>])
~~~

  * `MODULES`: (Required) The list of modules to wrap.
  * `WRAPPED_MODULES`: (Recommended) Not all modules are wrappable. This
    variable will be set to contain the list of modules which were wrapped.
  * `JAVA_OUTPUT`: Defaults to
    `${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/svtkJava`. Java source files are
    written to this directory. After generation, the files may be compiled as
    needed.
  * `LIBRARY_DESTINATION` (Recommended): If provided, dynamic loader
    information will be added to modules for loading dependent libraries.
  * `JNILIB_DESTINATION`: Where to install JNI libraries.
  * `JNILIB_COMPONENT`: Defaults to `jni`. The install component to use for JNI
    libraries.

For each wrapped module, a `<module>Java` target will be created. These targets
will have a `_svtk_module_java_files` property which is the list of generated
Java source files for that target.

For dependency purposes, the `<module>Java-java-sources` target may also be
used.
#]==]
function (svtk_module_wrap_java)
  cmake_parse_arguments(_svtk_java
    ""
    "JAVA_OUTPUT;WRAPPED_MODULES;LIBRARY_DESTINATION;JNILIB_DESTINATION;JNILIB_COMPONENT"
    "MODULES"
    ${ARGN})

  if (_svtk_java_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_wrap_java: "
      "${_svtk_java_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT _svtk_java_JAVA_OUTPUT)
    set(_svtk_java_JAVA_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/svtkJava")
  endif ()

  if (NOT _svtk_java_JNILIB_COMPONENT)
    set(_svtk_java_JNILIB_COMPONENT "jni")
  endif ()

  # Set up rpaths
  set(CMAKE_BUILD_RPATH_USE_ORIGIN 1)
  if (UNIX)
    if (APPLE)
      set(_svtk_java_origin_rpath_prefix
        "@loader_path")
    else ()
      set(_svtk_java_origin_rpath_prefix
        "$ORIGIN")
    endif ()

    list(APPEND CMAKE_INSTALL_RPATH
      # For sibling wrapped modules.
      "${_svtk_java_origin_rpath_prefix}")

    if (DEFINED _svtk_java_LIBRARY_DESTINATION AND DEFINED _svtk_java_JNILIB_DESTINATION)
      file(RELATIVE_PATH _svtk_java_relpath
        "/prefix/${_svtk_java_JNILIB_DESTINATION}"
        "/prefix/${_svtk_java_LIBRARY_DESTINATION}")

      list(APPEND CMAKE_INSTALL_RPATH
        # For libraries.
        "${_svtk_java_origin_rpath_prefix}/${_svtk_java_relpath}")
    endif ()
  endif ()

  if (DEFINED _svtk_java_JNILIB_DESTINATION)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_java_JNILIB_DESTINATION}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_java_JNILIB_DESTINATION}")
  endif ()

  if (NOT _svtk_java_MODULES)
    message(WARNING
      "No modules were requested for java wrapping.")
    return ()
  endif ()

  set(_svtk_java_all_wrapped_modules)
  foreach (_svtk_java_module IN LISTS _svtk_java_MODULES)
    _svtk_module_get_module_property("${_svtk_java_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_java_exclude_wrap)
    _svtk_module_get_module_property("${_svtk_java_module}"
      PROPERTY  "library_name"
      VARIABLE  _svtk_java_library_name)
    _svtk_module_wrap_java_library("${_svtk_java_library_name}" "${_svtk_java_module}")

    if (TARGET "${_svtk_java_library_name}Java")
      list(APPEND _svtk_java_all_wrapped_modules
        "${_svtk_java_module}")
    endif ()
  endforeach ()

  if (NOT _svtk_java_all_wrapped_modules)
    message(FATAL_ERROR
      "No modules given could be wrapped.")
  endif ()

  if (DEFINED _svtk_java_WRAPPED_MODULES)
    set("${_svtk_java_WRAPPED_MODULES}"
      "${_svtk_java_all_wrapped_modules}"
      PARENT_SCOPE)
  endif ()
endfunction ()

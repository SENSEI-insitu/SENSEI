#[==[
@defgroup module Module CMake APIs
@defgroup module-internal Module Internal CMake APIs
@defgroup module-impl Module Implementation CMake APIs
@defgroup module-support Module Support CMake APIs
#]==]

#[==[
@ingroup module
@page module-api-overview Module API

This module includes functions to find and build SVTK modules. A module is a set
of related functionality. These are then compiled together into libraries at
the "kit" level. Each module may be enabled or disabled individually and its
dependencies will be built as needed.

All functions strictly check their arguments. Any unrecognized or invalid
values for a function cause errors to be raised.
#]==]

#[==[
@ingroup module-internal
@page module-internal-api Internal API

The SVTK module system provides some API functions for use by other code which
consumes SVTK modules (primarily language wrappers). This file documents these
APIs. They may start with `_svtk_module`, but they are intended for use in cases
of language wrappers or dealing with trickier third party packages.
#]==]

#[==[
@ingroup module-impl
@page module-impl-api Implementation API

These functions are purely internal implementation details. No guarantees are
made for them and they may change at any time (including wrapping code calls).
Note that these functions are usually very lax in their argument parsing.
#]==]

#[==[
@ingroup module-internal
@brief Conditionally output debug statements

The @ref _svtk_module_debug function is provided to assist in debugging. It is
controlled by the `_svtk_module_log` variable which contains a list of "domains"
to debug.

~~~
_svtk_module_debug(<domain> <format>)
~~~

If the `domain` is enabled for debugging, the `format` argument is configured
and printed. It should contain `@` variable expansions to replace rather than
it being done outside. This helps to avoid the cost of generating large strings
when debugging is disabled.
#]==]
function (_svtk_module_debug domain format)
  if (NOT _svtk_module_log STREQUAL "ALL" AND
      NOT domain IN_LIST _svtk_module_log)
    return ()
  endif ()

  string(CONFIGURE "${format}" _svtk_module_debug_msg)
  if (_svtk_module_debug_msg)
    message(STATUS
      "SVTK module debug ${domain}: ${_svtk_module_debug_msg}")
  endif ()
endfunction ()

# TODO: Support finding `svtk.module` and `svtk.kit` contents in the
# `CMakeLists.txt` files for the module via a comment header.

#[==[
@ingroup module
@brief Find `svtk.kit` files in a set of directories

~~~
svtk_module_find_kits(<output> [<directory>...])
~~~

This scans the given directories recursively for `svtk.kit` files and put the
paths into the output variable.
#]==]
function (svtk_module_find_kits output)
  set(_svtk_find_kits_all)
  foreach (_svtk_find_kits_directory IN LISTS ARGN)
    file(GLOB_RECURSE _svtk_find_kits_kits
      "${_svtk_find_kits_directory}/svtk.kit")
    list(APPEND _svtk_find_kits_all
      ${_svtk_find_kits_kits})
  endforeach ()
  set("${output}" ${_svtk_find_kits_all} PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module
@brief Find `svtk.module` files in a set of directories

~~~
svtk_module_find_modules(<output> [<directory>...])
~~~

This scans the given directories recursively for `svtk.module` files and put the
paths into the output variable. Note that module files are assumed to live next
to the `CMakeLists.txt` file which will build the module.
#]==]
function (svtk_module_find_modules output)
  set(_svtk_find_modules_all)
  foreach (_svtk_find_modules_directory IN LISTS ARGN)
    file(GLOB_RECURSE _svtk_find_modules_modules
      "${_svtk_find_modules_directory}/svtk.module")
    list(APPEND _svtk_find_modules_all
      ${_svtk_find_modules_modules})
  endforeach ()
  set("${output}" ${_svtk_find_modules_all} PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module-internal
@brief Split a module name into a namespace and target component

Module names may include a namespace. This function splits the name into a
namespace and target name part.

~~~
_svtk_module_split_module_name(<name> <prefix>)
~~~

The `<prefix>_NAMESPACE` and `<prefix>_TARGET_NAME` variables will be set in
the calling scope.
#]==]
function (_svtk_module_split_module_name name prefix)
  string(FIND "${name}" "::" namespace_pos)
  if (namespace_pos EQUAL -1)
    set(namespace "")
    set(target_name "${name}")
  else ()
    string(SUBSTRING "${name}" 0 "${namespace_pos}" namespace)
    math(EXPR name_pos "${namespace_pos} + 2")
    string(SUBSTRING "${name}" "${name_pos}" -1 target_name)
  endif ()

  set("${prefix}_NAMESPACE"
    "${namespace}"
    PARENT_SCOPE)
  set("${prefix}_TARGET_NAME"
    "${target_name}"
    PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module
@page module-overview Module overview

@section module-parse-module svtk.module file contents

The `svtk.module` file is parsed and used as arguments to a CMake function which
stores information about the module for use when building it. Note that no
variable expansion is allowed and it is not CMake code, so no control flow is
allowed. Comments are supported and any content after a `#` on a line is
treated as a comment. Due to the breakdown of the content, quotes are not
meaningful within the files.

Example:

~~~
NAME
  SVTK::CommonCore
LIBRARY_NAME
  svtkCommonCore
DESCRIPTION
  The base SVTK library.
GROUPS
  StandAlone
DEPENDS
  SVTK::kwiml
PRIVATE_DEPENDS
  SVTK::svtksys
  SVTK::utf8
~~~

All values are optional unless otherwise noted. The following arguments are
supported:

  * `NAME`: (Required) The name of the module.
  * `LIBRARY_NAME`: The base name of the library file. It defaults to the
    module name, but any namespaces are removed. For example, a `NS::Foo`
    module will have a default `LIBRARY_NAME` of `Foo`.
  * `DESCRIPTION`: (Recommended) Short text describing what the module is for.
  * `KIT`: The name of the kit the module belongs to (see `Kits files` for more
    information).
  * `IMPLEMENTABLE`: If present, the module contains logic which supports the
    autoinit functionality.
  * `GROUPS`: Modules may belong to "groups" which is exposed as a build
    option. This allows for enabling a set of modules with a single build
    option.
  * `CONDITION`: Arguments to CMake's `if` command which may be used to hide
    the module for certain platforms or other reasons. If the expression is
    false, the module is completely ignored.
  * `DEPENDS`: A list of modules which are required by this module and modules
    using this module.
  * `PRIVATE_DEPENDS`: A list of modules which are required by this module, but
    not by those using this module.
  * `OPTIONAL_DEPENDS`: A list of modules which are used by this module if
    enabled; these are treated as `PRIVATE_DEPENDS` if they exist.
  * `ORDER_DEPENDS`: Dependencies which only matter for ordering. This does not
    mean that the module will be enabled, just guaranteed to build before this
    module.
  * `IMPLEMENTS`: A list of modules for which this module needs to register
    with.
  * `TEST_DEPENDS`: Modules required by the test suite for this module.
  * `TEST_OPTIONAL_DEPENDS`: Modules used by the test suite for this module if
    available.
  * `TEST_LABELS`: Labels to apply to the tests of this module. By default, the
    module name is applied as a label.
  * `EXCLUDE_WRAP`: If present, this module should not be wrapped in any
    language.
  * `THIRD_PARTY`: If present, this module is a third party module.
#]==]

#[==[
@ingroup module-impl
@brief Parse `svtk.module` file contents

This macro places all `svtk.module` keyword "arguments" into the caller's scope
prefixed with the value of `name_output` which is set to the `NAME` of the
module.

~~~
_svtk_module_parse_module_args(name_output <svtk.module args...>)
~~~

For example, this `svtk.module` file:

~~~
NAME
  Namespace::Target
LIBRARY_NAME
  nsTarget
~~~

called with `_svtk_module_parse_module_args(name ...)` will set the following
variables in the calling scope:

  - `name`: `Namespace::Target`
  - `Namespace::Target_LIBRARY_NAME`: `nsTarget`

With namespace support for module names, the variable should instead be
referenced via `${${name}_LIBRARY_NAME}` instead.
#]==]
macro (_svtk_module_parse_module_args name_output)
  cmake_parse_arguments("_name"
    ""
    "NAME"
    ""
    ${ARGN})

  if (NOT _name_NAME)
    message(FATAL_ERROR
      "A SVTK module requires a name (from ${_svtk_scan_module_file}).")
  endif ()
  set("${name_output}" "${_name_NAME}")

  cmake_parse_arguments("${_name_NAME}"
    "IMPLEMENTABLE;EXCLUDE_WRAP;THIRD_PARTY"
    "LIBRARY_NAME;NAME;KIT"
    "GROUPS;DEPENDS;PRIVATE_DEPENDS;OPTIONAL_DEPENDS;ORDER_DEPENDS;TEST_DEPENDS;TEST_OPTIONAL_DEPENDS;TEST_LABELS;DESCRIPTION;CONDITION;IMPLEMENTS"
    ${ARGN})

  if (${_name_NAME}_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for ${_name_NAME}: "
      "${${_name_NAME}_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT ${_name_NAME}_DESCRIPTION AND _svtk_module_warnings)
    message(WARNING "The ${_name_NAME} module should have a description")
  endif ()
  string(REPLACE ";" " " "${_name_NAME}_DESCRIPTION" "${${_name_NAME}_DESCRIPTION}")

  _svtk_module_split_module_name("${_name_NAME}" "${_name_NAME}")

  if (NOT DEFINED "${_name_NAME}_LIBRARY_NAME")
    set("${_name_NAME}_LIBRARY_NAME" "${${_name_NAME}_TARGET_NAME}")
  endif ()

  if (NOT ${_name_NAME}_LIBRARY_NAME)
    message(FATAL_ERROR "The ${_name_NAME} module must have a non-empty `LIBRARY_NAME`.")
  endif ()

  list(APPEND "${_name_NAME}_TEST_LABELS"
    "${${_name_NAME}_NAME}"
    "${${_name_NAME}_LIBRARY_NAME}")
endmacro ()

#[==[
@page module-overview

@section module-parse-kit svtk.kit file contents

The `svtk.kit` file is parsed similarly to `svtk.module` files. Kits are intended
to bring together related modules into a single library in order to reduce the
number of objects that linkers need to deal with.

Example:

~~~
NAME
  SVTK::Common
LIBRARY_NAME
  svtkCommon
DESCRIPTION
  Core utilities for SVTK.
~~~

All values are optional unless otherwise noted. The following arguments are
supported:

  * `NAME`: (Required) The name of the kit.
  * `LIBRARY_NAME`: The base name of the library file. It defaults to the
    module name, but any namespaces are removed. For example, a `NS::Foo`
    module will have a default `LIBRARY_NAME` of `Foo`.
  * `DESCRIPTION`: (Recommended) Short text describing what the kit contains.
#]==]

#[==[
@ingroup module-impl
@brief Parse `svtk.kit` file contents

Just like @ref _svtk_module_parse_module_args, but for kits.
#]==]
macro (_svtk_module_parse_kit_args name_output)
  cmake_parse_arguments("_name"
    ""
    "NAME"
    ""
    ${ARGN})

  if (NOT _name_NAME)
    message(FATAL_ERROR
      "A SVTK kit requires a name (from ${_svtk_scan_kit_file}).")
  endif ()
  set("${name_output}" "${_name_NAME}")

  cmake_parse_arguments("${_name_NAME}"
    ""
    "NAME;LIBRARY_NAME"
    "DESCRIPTION"
    ${ARGN})

  if (${_name_NAME}_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for ${_name_NAME}: "
      "${${_name_NAME}_UNPARSED_ARGUMENTS}")
  endif ()

  _svtk_module_split_module_name("${_name_NAME}" "${_name_NAME}")

  if (NOT DEFINED "${_name_NAME}_LIBRARY_NAME")
    set("${_name_NAME}_LIBRARY_NAME" "${${_name_NAME}_TARGET_NAME}")
  endif ()

  if (NOT ${_name_NAME}_LIBRARY_NAME)
    message(FATAL_ERROR "The ${_name_NAME} module must have a non-empty `LIBRARY_NAME`.")
  endif ()

  if (NOT ${_name_NAME}_DESCRIPTION AND _svtk_module_warnings)
    message(WARNING "The ${_name_NAME} kit should have a description")
  endif ()
  string(REPLACE ";" " " "${_name_NAME}_DESCRIPTION" "${${_name_NAME}_DESCRIPTION}")
endmacro ()

#[==[
@page module-overview

@ingroup module
@section module-enable-status Enable status values

Modules and groups are enable and disable preferences are specified using a
5-way flag setting:

  - `YES`: The module or group must be built.
  - `NO`: The module or group must not be built.
  - `WANT`: The module or group should be built if possible.
  - `DONT_WANT`: The module or group should only be built if required (e.g.,
    via a dependency).
  - `DEFAULT`: Acts as either `WANT` or `DONT_WANT` based on the group settings
    for the module or `WANT_BY_DEFAULT` option to @ref svtk_module_scan if no
    other preference is specified. This is usually handled via another setting
    in the main project.

If a `YES` module preference requires a module with a `NO` preference, an error
is raised.

A module with a setting of `DEFAULT` will look for its first non-`DEFAULT`
group setting and only if all of those are set to `DEFAULT` is the
`WANT_BY_DEFAULT` setting used.
#]==]

#[==[
@ingroup module-impl
@brief Verify enable values

Verifies that the variable named as the first parameter is a valid `enable
status` value.

~~~
_svtk_module_verify_enable_value(var)
~~~
#]==]
function (_svtk_module_verify_enable_value var)
  if (NOT (${var} STREQUAL "YES" OR
           ${var} STREQUAL "WANT" OR
           ${var} STREQUAL "DONT_WANT" OR
           ${var} STREQUAL "NO" OR
           ${var} STREQUAL "DEFAULT"))
    message(FATAL_ERROR
      "The `${var}` variable must be one of `YES`, `WANT`, `DONT_WANT`, `NO`, "
      "or `DEFAULT`. Found `${${var}}`.")
  endif ()
endfunction ()

include("${CMAKE_CURRENT_LIST_DIR}/svtkTopologicalSort.cmake")

#[==[
@ingroup module
@brief Scan modules and kits

Once all of the modules and kits files have been found, they are "scanned" to
determine what modules are enabled or required.

~~~
svtk_module_scan(
  MODULE_FILES              <file>...
  [KIT_FILES                <file>...]
  PROVIDES_MODULES          <variable>
  [PROVIDES_KITS            <variable>]
  [REQUIRES_MODULES         <variable>]
  [REQUEST_MODULES          <module>...]
  [REJECT_MODULES           <module>...]
  [UNRECOGNIZED_MODULES     <variable>]
  [WANT_BY_DEFAULT          <ON|OFF>]
  [HIDE_MODULES_FROM_CACHE  <ON|OFF>]
  [ENABLE_TESTS             <ON|OFF|WANT|DEFAULT>])
~~~

The `MODULE_FILES` and `PROVIDES_MODULES` arguments are required. Modules which
refer to kits must be scanned at the same time as their kits. This is so that
modules may not add themselves to kits declared prior. The arguments are as follows:

  * `MODULE_FILES`: (Required) The list of module files to scan.
  * `KIT_FILES`: The list of kit files to scan.
  * `PROVIDES_MODULES`: (Required) This variable will contain the list of
    modules which are enabled due to this scan.
  * `PROVIDES_KITS`: (Required if `KIT_FILES` are provided) This variable will
    contain the list of kits which are enabled due to this scan.
  * `REQUIRES_MODULES`: This variable will contain the list of modules required
    by the enabled modules that were not scanned.
  * `REQUEST_MODULES`: The list of modules required by previous scans.
  * `REJECT_MODULES`: The list of modules to exclude from the scan. If any of
    these modules are required, an error will be raised.
  * `UNRECOGNIZED_MODULES`: This variable will contain the list of requested
    modules that were not scanned.
  * `WANT_BY_DEFAULT`: (Defaults to `OFF`) Whether modules should default to
    being built or not.
  * `HIDE_MODULES_FROM_CACHE`: (Defaults to `OFF`) Whether or not to hide the
    control variables from the cache or not. If enabled, modules will not be
    built unless they are required elsewhere.
  * `ENABLE_TESTS`: (Defaults to `WANT`) Whether or not modules required by
    the tests for the scanned modules should be enabled or not.
    - `ON`: Modules listed as `TEST_DEPENDS` will be required.
    - `OFF`: Test modules will not be considered.
    - `WANT`: Test dependencies will enable modules if possible.
    - `DEFAULT`: Test modules will be enabled if their required dependencies
      are satisfied and skipped otherwise.

@section module-scanning-multiple Scanning multiple groups of modules

When scanning complicated projects, multiple scans may be required to get
defaults set properly. The `REQUIRES_MODULES`, `REQUEST_MODULES`, and
`UNRECOGNIZED_MODULES` arguments are meant to deal with this case. As an
example, imagine a project with its source code, third party dependencies, as
well as some utility modules which should only be built as necessary. Here, the
project would perform three scans, one for each "grouping" of modules:

~~~{.cmake}
# Scan our modules first because we need to know what of the other groups we
# need.
svtk_module_find_modules(our_modules "${CMAKE_CURRENT_SOURCE_DIR}/src")
svtk_module_scan(
  MODULE_FILES      ${our_modules}
  PROVIDES_MODULES  our_enabled_modules
  REQUIRES_MODULES  required_modules)

# Scan the third party modules, requesting only those that are necessary, but
# allowing them to be toggled during the build.
svtk_module_find_modules(third_party_modules "${CMAKE_CURRENT_SOURCE_DIR}/third-party")
svtk_module_scan(
  MODULE_FILES            ${third_party_modules}
  PROVIDES_MODULES        third_party_enabled_modules
  # These modules were requested by an earlier scan.
  REQUEST_MODULES         ${required_modules}
  REQUIRES_MODULES        required_modules
  UNRECOGNIZED_MODULES    unrecognized_modules)

# These modules are internal and should only be built if necessary. There is no
# need to support them being enabled independently, so hide them from the
# cache.
svtk_module_find_modules(utility_modules "${CMAKE_CURRENT_SOURCE_DIR}/utilities")
svtk_module_scan(
  MODULE_FILES            ${utility_modules}
  PROVIDES_MODULES        utility_enabled_modules
  # These modules were either requested or unrecognized by an earlier scan.
  REQUEST_MODULES         ${required_modules}
                          ${unrecognized_modules}
  REQUIRES_MODULES        required_modules
  UNRECOGNIZED_MODULES    unrecognized_modules
  HIDE_MODULES_FROM_CACHE ON)

if (required_modules OR unrecognized_modules)
  # Not all of the modules we required were found. This should probably error out.
endif ()
~~~
#]==]
function (svtk_module_scan)
  cmake_parse_arguments(_svtk_scan
    ""
    "WANT_BY_DEFAULT;HIDE_MODULES_FROM_CACHE;PROVIDES_MODULES;REQUIRES_MODULES;UNRECOGNIZED_MODULES;ENABLE_TESTS;PROVIDES_KITS"
    "MODULE_FILES;KIT_FILES;REQUEST_MODULES;REJECT_MODULES"
    ${ARGN})

  if (_svtk_scan_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_scan: "
      "${_svtk_scan_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_scan_WANT_BY_DEFAULT)
    set(_svtk_scan_WANT_BY_DEFAULT OFF)
  endif ()

  if (NOT DEFINED _svtk_scan_HIDE_MODULES_FROM_CACHE)
    set(_svtk_scan_HIDE_MODULES_FROM_CACHE OFF)
  endif ()

  if (NOT DEFINED _svtk_scan_PROVIDES_MODULES)
    message(FATAL_ERROR
      "The `PROVIDES_MODULES` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_scan_PROVIDES_KITS AND _svtk_scan_KIT_FILES)
    message(FATAL_ERROR
      "The `PROVIDES_KITS` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_scan_ENABLE_TESTS)
    set(_svtk_scan_ENABLE_TESTS "WANT")
  endif ()

  if (NOT (_svtk_scan_ENABLE_TESTS STREQUAL "ON" OR
           _svtk_scan_ENABLE_TESTS STREQUAL "OFF" OR
           _svtk_scan_ENABLE_TESTS STREQUAL "WANT" OR
           _svtk_scan_ENABLE_TESTS STREQUAL "DEFAULT"))
    message(FATAL_ERROR
      "The `ENABLE_TESTS` argument must be one of `ON`, `OFF`, `WANT`, or "
      "`DEFAULT`. " "Received `${_svtk_scan_ENABLE_TESTS}`.")
  endif ()

  if (NOT _svtk_scan_MODULE_FILES)
    message(FATAL_ERROR
      "No module files given to scan.")
  endif ()

  set(_svtk_scan_option_default_type STRING)
  if (_svtk_scan_HIDE_MODULES_FROM_CACHE)
    set(_svtk_scan_option_default_type INTERNAL)
  endif ()

  set(_svtk_scan_all_kits)

  foreach (_svtk_scan_kit_file IN LISTS _svtk_scan_KIT_FILES)
    if (NOT IS_ABSOLUTE "${_svtk_scan_kit_file}")
      set(_svtk_scan_kit_file "${CMAKE_CURRENT_SOURCE_DIR}/${_svtk_scan_kit_file}")
    endif ()
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND
      PROPERTY
        CMAKE_CONFIGURE_DEPENDS "${_svtk_scan_kit_file}")

    file(READ "${_svtk_scan_kit_file}" _svtk_scan_kit_args)
    # Replace comments.
    string(REGEX REPLACE "#[^\n]*\n" "\n" _svtk_scan_kit_args "${_svtk_scan_kit_args}")
    # Use argument splitting.
    string(REGEX REPLACE "( |\n)+" ";" _svtk_scan_kit_args "${_svtk_scan_kit_args}")
    _svtk_module_parse_kit_args(_svtk_scan_kit_name ${_svtk_scan_kit_args})
    _svtk_module_debug(kit "@_svtk_scan_kit_name@ declared by @_svtk_scan_kit_file@")

    list(APPEND _svtk_scan_all_kits
      "${_svtk_scan_kit_name}")

    # Set properties for building.
    set_property(GLOBAL
      PROPERTY
        "_svtk_kit_${_svtk_scan_kit_name}_namespace" "${${_svtk_scan_kit_name}_NAMESPACE}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_kit_${_svtk_scan_kit_name}_target_name" "${${_svtk_scan_kit_name}_TARGET_NAME}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_kit_${_svtk_scan_kit_name}_library_name" "${${_svtk_scan_kit_name}_LIBRARY_NAME}")
  endforeach ()

  set(_svtk_scan_all_modules)
  set(_svtk_scan_all_groups)
  set(_svtk_scan_rejected_modules)

  # Read all of the module files passed in.
  foreach (_svtk_scan_module_file IN LISTS _svtk_scan_MODULE_FILES)
    if (NOT IS_ABSOLUTE "${_svtk_scan_module_file}")
      set(_svtk_scan_module_file "${CMAKE_CURRENT_SOURCE_DIR}/${_svtk_scan_module_file}")
    endif ()
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND
      PROPERTY
        CMAKE_CONFIGURE_DEPENDS "${_svtk_scan_module_file}")

    file(READ "${_svtk_scan_module_file}" _svtk_scan_module_args)
    # Replace comments.
    string(REGEX REPLACE "#[^\n]*\n" "\n" _svtk_scan_module_args "${_svtk_scan_module_args}")
    # Use argument splitting.
    string(REGEX REPLACE "( |\n)+" ";" _svtk_scan_module_args "${_svtk_scan_module_args}")
    _svtk_module_parse_module_args(_svtk_scan_module_name ${_svtk_scan_module_args})
    _svtk_module_debug(module "@_svtk_scan_module_name@ declared by @_svtk_scan_module_file@")
    string(REPLACE "::" "_" _svtk_scan_module_name_safe "${_svtk_scan_module_name}")

    if (${_svtk_scan_module_name}_THIRD_PARTY)
      if (_svtk_module_warnings)
        if (${_svtk_scan_module_name}_EXCLUDE_WRAP)
          message(WARNING
            "The third party ${_svtk_scan_module_name} module does not need to "
            "declare `EXCLUDE_WRAP` also.")
        endif ()
      endif ()
      if (${_svtk_scan_module_name}_IMPLEMENTABLE)
        message(FATAL_ERROR
          "The third party ${_svtk_scan_module_name} module may not be "
          "`IMPLEMENTABLE`.")
      endif ()
      if (${_svtk_scan_module_name}_IMPLEMENTS)
        message(FATAL_ERROR
          "The third party ${_svtk_scan_module_name} module may not "
          "`IMPLEMENTS` another module.")
      endif ()
      if (${_svtk_scan_module_name}_KIT)
        message(FATAL_ERROR
          "The third party ${_svtk_scan_module_name} module may not be part of "
          "a kit (${${_svtk_scan_module_name}_KIT}).")
      endif ()
    endif ()

    if (${_svtk_scan_module_name}_KIT)
      if (NOT ${_svtk_scan_module_name}_KIT IN_LIST _svtk_scan_all_kits)
        message(FATAL_ERROR
          "The ${_svtk_scan_module_name} belongs to the "
          "${${_svtk_scan_module_name}_KIT} kit, but it has not been scanned.")
      endif ()
    endif ()

    # Check if the module is visible. Modules which have a failing condition
    # are basically invisible.
    if (DEFINED ${_svtk_scan_module_name}_CONDITION)
      if (NOT (${${_svtk_scan_module_name}_CONDITION}))
        if (DEFINED "SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}")
          set_property(CACHE "SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}"
            PROPERTY
              TYPE INTERNAL)
        endif ()
        _svtk_module_debug(module "@_svtk_scan_module_name@ hidden by its `CONDITION`")
        continue ()
      endif ()
    endif ()

    # Determine whether we should provide a user-visible option for this
    # module.
    set(_svtk_build_use_option 1)
    if (DEFINED _svtk_scan_REQUEST_MODULE)
      if (_svtk_scan_module_name IN_LIST _svtk_scan_REQUEST_MODULE)
        set("_svtk_scan_enable_${_svtk_scan_module_name}" YES)
        set(_svtk_build_use_option 0)
      endif ()
    endif ()
    if (DEFINED _svtk_scan_REJECT_MODULES)
      if (_svtk_scan_module_name IN_LIST _svtk_scan_REJECT_MODULES)
        if (NOT _svtk_build_use_option)
          message(FATAL_ERROR
            "The ${_svtk_scan_module_name} module has been requested and rejected.")
        endif ()
        # Rejected modules should not have a build option.
        set(_svtk_build_use_option 0)
        list(APPEND _svtk_scan_rejected_modules
          "${_svtk_scan_module_name}")
      endif ()
    endif ()

    # Handle cache entries and determine the enabled state of the module from
    # the relevant cache variables.
    if (_svtk_build_use_option)
      set("SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}" "DEFAULT"
        CACHE STRING "Enable the ${_svtk_scan_module_name} module. ${${_svtk_scan_module_name}_DESCRIPTION}")
      mark_as_advanced("SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}")
      set_property(CACHE "SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}"
        PROPERTY
          STRINGS "YES;WANT;DONT_WANT;NO;DEFAULT")
      _svtk_module_verify_enable_value("SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}")

      if (NOT SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe} STREQUAL "DEFAULT")
        set("_svtk_scan_enable_${_svtk_scan_module_name}" "${SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}}")
        _svtk_module_debug(enable "@_svtk_scan_module_name@ is `${_svtk_scan_enable_${_svtk_scan_module_name}}` by cache value")
      endif ()

      # Check the state of any groups the module belongs to.
      foreach (_svtk_scan_group IN LISTS "${_svtk_scan_module_name}_GROUPS")
        if (NOT DEFINED "SVTK_GROUP_ENABLE_${_svtk_scan_group}")
          set(_svtk_scan_group_default "DEFAULT")
          if (DEFINED "_svtk_module_group_default_${_svtk_scan_group}")
            set(_svtk_scan_group_default "${_svtk_module_group_default_${_svtk_scan_group}}")
          endif ()
          set("SVTK_GROUP_ENABLE_${_svtk_scan_group}" "${_svtk_scan_group_default}"
            CACHE STRING "Enable the ${_svtk_scan_group} group modules.")
          set_property(CACHE "SVTK_GROUP_ENABLE_${_svtk_scan_group}"
            PROPERTY
              STRINGS "YES;WANT;DONT_WANT;NO;DEFAULT")
          set_property(CACHE "SVTK_GROUP_ENABLE_${_svtk_scan_group}"
            PROPERTY
              TYPE "${_svtk_scan_option_default_type}")
        endif ()
        _svtk_module_verify_enable_value("SVTK_GROUP_ENABLE_${_svtk_scan_group}")

        if (NOT SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe} STREQUAL "DEFAULT")
          continue ()
        endif ()

        # Determine the state of the group.
        set(_svtk_scan_group_enable "${SVTK_GROUP_ENABLE_${_svtk_scan_group}}")
        if (NOT _svtk_scan_group_enable STREQUAL "DEFAULT")
          set("_svtk_scan_enable_${_svtk_scan_module_name}" "${_svtk_scan_group_enable}")
          _svtk_module_debug(enable "@_svtk_scan_module_name@ is DEFAULT, using group `@_svtk_scan_group@` setting: @_svtk_scan_group_enable@")
        endif ()
      endforeach ()

      set_property(CACHE "SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe}"
        PROPERTY
          TYPE "${_svtk_scan_option_default_type}")
    endif ()

    if (NOT DEFINED "_svtk_scan_enable_${_svtk_scan_module_name}" AND
        SVTK_MODULE_ENABLE_${_svtk_scan_module_name_safe} STREQUAL "DEFAULT")
      if (_svtk_scan_WANT_BY_DEFAULT)
        set("_svtk_scan_enable_${_svtk_scan_module_name}" "WANT")
      else ()
        set("_svtk_scan_enable_${_svtk_scan_module_name}" "DONT_WANT")
      endif ()
      _svtk_module_debug(enable "@_svtk_scan_module_name@ is DEFAULT, using `WANT_BY_DEFAULT`: ${_svtk_scan_enable_${_svtk_scan_module_name}}")
    endif ()

    list(APPEND _svtk_scan_all_modules
      "${_svtk_scan_module_name}")
    set("_svtk_scan_${_svtk_scan_module_name}_all_depends"
      ${${_svtk_scan_module_name}_DEPENDS}
      ${${_svtk_scan_module_name}_PRIVATE_DEPENDS})

    if (${_svtk_scan_module_name}_THIRD_PARTY)
      set("${_svtk_scan_module_name}_EXCLUDE_WRAP" TRUE)
      set("${_svtk_scan_module_name}_IMPLEMENTABLE" FALSE)
      set("${_svtk_scan_module_name}_IMPLEMENTS")
    endif ()

    if (${_svtk_scan_module_name}_KIT)
      _svtk_module_debug(kit "@_svtk_scan_module_name@ belongs to the ${${_svtk_scan_module_name}_KIT} kit")
    endif ()

    # Set properties for building.
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_file" "${_svtk_scan_module_file}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_namespace" "${${_svtk_scan_module_name}_NAMESPACE}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_target_name" "${${_svtk_scan_module_name}_TARGET_NAME}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_library_name" "${${_svtk_scan_module_name}_LIBRARY_NAME}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_third_party" "${${_svtk_scan_module_name}_THIRD_PARTY}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_exclude_wrap" "${${_svtk_scan_module_name}_EXCLUDE_WRAP}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_kit" "${${_svtk_scan_module_name}_KIT}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_depends" "${${_svtk_scan_module_name}_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_order_depends" "${${_svtk_scan_module_name}_ORDER_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_private_depends" "${${_svtk_scan_module_name}_PRIVATE_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_optional_depends" "${${_svtk_scan_module_name}_OPTIONAL_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_test_depends" "${${_svtk_scan_module_name}_TEST_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_test_optional_depends" "${${_svtk_scan_module_name}_TEST_OPTIONAL_DEPENDS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_test_labels" "${${_svtk_scan_module_name}_TEST_LABELS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_implements" "${${_svtk_scan_module_name}_IMPLEMENTS}")
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_${_svtk_scan_module_name}_implementable" "${${_svtk_scan_module_name}_IMPLEMENTABLE}")
  endforeach ()

  set(_svtk_scan_current_modules "${_svtk_scan_all_modules}")
  svtk_topological_sort(_svtk_scan_all_modules "_svtk_scan_" "_all_depends")

  set(_svtk_scan_provided_modules)
  set(_svtk_scan_required_modules)
  set(_svtk_scan_disabled_modules)

  # Seed the `_svtk_scan_provide_` variables with modules requested and rejected
  # as arguments.
  foreach (_svtk_scan_request_module IN LISTS _svtk_scan_REQUEST_MODULES)
    set("_svtk_scan_provide_${_svtk_scan_request_module}" ON)
    _svtk_module_debug(provide "@_svtk_scan_request_module@ is provided via `REQUEST_MODULES`")
  endforeach ()
  foreach (_svtk_scan_reject_module IN LISTS _svtk_scan_REJECT_MODULES)
    set("_svtk_scan_provide_${_svtk_scan_reject_module}" OFF)
    _svtk_module_debug(provide "@_svtk_scan_reject_module@ is not provided via `REJECT_MODULES`")
  endforeach ()

  # Traverse the graph classifying the quad-state for enabling modules into a
  # boolean stored in the `_svtk_scan_provide_` variables.
  foreach (_svtk_scan_module IN LISTS _svtk_scan_all_modules)
    if (NOT _svtk_scan_module IN_LIST _svtk_scan_current_modules)
      _svtk_module_debug(provide "@_svtk_scan_module@ is ignored because it is not in the current scan set")
      continue ()
    endif ()

    if (DEFINED "_svtk_scan_provide_${_svtk_scan_module}")
      # Already done.
    elseif (_svtk_scan_enable_${_svtk_scan_module} STREQUAL "YES")
      # Mark enabled modules as to-be-provided. Any errors with requiring a
      # disabled module will be dealt with later.
      set("_svtk_scan_provide_${_svtk_scan_module}" ON)
      _svtk_module_debug(provide "@_svtk_scan_module@ is provided due to `YES` setting")
    elseif (_svtk_scan_enable_${_svtk_scan_module} STREQUAL "WANT")
      # Check to see if we can provide this module by checking of any of its
      # dependencies have been disabled.
      set(_svtk_scan_test_depends)
      if (NOT ${_svtk_scan_module}_THIRD_PARTY AND _svtk_scan_ENABLE_TESTS STREQUAL "ON")
        # If the tests have to be on, we also need the test dependencies.
        set(_svtk_scan_test_depends "${${_svtk_scan_module}_TEST_DEPENDS}")
      endif ()

      set("_svtk_scan_provide_${_svtk_scan_module}" ON)
      _svtk_module_debug(provide "@_svtk_scan_module@ is provided due to `WANT` setting")
      foreach (_svtk_scan_module_depend IN LISTS "${_svtk_scan_module}_DEPENDS" "${_svtk_scan_module}_PRIVATE_DEPENDS" _svtk_scan_test_depends)
        if (DEFINED "_svtk_scan_provide_${_svtk_scan_module_depend}" AND NOT _svtk_scan_provide_${_svtk_scan_module_depend})
          set("_svtk_scan_provide_${_svtk_scan_module}" OFF)
          _svtk_module_debug(provide "@_svtk_scan_module@ is not provided due to not provided dependency @_svtk_scan_module_depend@")
          break ()
        endif ()
      endforeach ()
    elseif (_svtk_scan_enable_${_svtk_scan_module} STREQUAL "DONT_WANT")
      # Check for disabled dependencies and disable if so.
      foreach (_svtk_scan_module_depend IN LISTS "${_svtk_scan_module}_DEPENDS" "${_svtk_scan_module}_PRIVATE_DEPENDS" _svtk_scan_test_depends)
        if (DEFINED "_svtk_scan_provide_${_svtk_scan_module_depend}" AND NOT _svtk_scan_provide_${_svtk_scan_module_depend})
          set("_svtk_scan_provide_${_svtk_scan_module}" OFF)
          _svtk_module_debug(provide "@_svtk_scan_module@ is not provided due to not provided dependency @_svtk_scan_module_depend@")
          break ()
        endif ()
      endforeach ()
    elseif (_svtk_scan_enable_${_svtk_scan_module} STREQUAL "NO")
      # Disable the module.
      set("_svtk_scan_provide_${_svtk_scan_module}" OFF)
      _svtk_module_debug(provide "@_svtk_scan_module@ is not provided due to `NO` setting")
    endif ()

    # Collect disabled modules into a list.
    if (DEFINED "_svtk_scan_provide_${_svtk_scan_module}" AND NOT _svtk_scan_provide_${_svtk_scan_module})
      list(APPEND _svtk_scan_disabled_modules
        "${_svtk_scan_module}")
    endif ()

    if (NOT DEFINED "_svtk_scan_provide_${_svtk_scan_module}")
      _svtk_module_debug(provide "@_svtk_scan_module@ is indeterminite (${_svtk_scan_enable_${_svtk_scan_module}})")
    endif ()
  endforeach ()

  # Scan all modules from the top of tree to the bottom.
  list(REVERSE _svtk_scan_all_modules)
  foreach (_svtk_scan_module IN LISTS _svtk_scan_all_modules)
    if (NOT _svtk_scan_module IN_LIST _svtk_scan_current_modules)
      continue ()
    endif ()

    # If we're providing this module...
    if (_svtk_scan_provide_${_svtk_scan_module})
      list(APPEND _svtk_scan_provided_modules
        "${_svtk_scan_module}")

      # Grab any test dependencies that are required.
      set(_svtk_scan_test_depends)
      set(_svtk_scan_test_wants)
      if (NOT ${_svtk_scan_module}_THIRD_PARTY)
        if (_svtk_scan_ENABLE_TESTS STREQUAL "ON")
          set_property(GLOBAL APPEND
            PROPERTY
              "_svtk_module_test_modules" "${_svtk_scan_module}")
          set(_svtk_scan_test_depends "${${_svtk_scan_module}_TEST_DEPENDS}")
        elseif (_svtk_scan_ENABLE_TESTS STREQUAL "WANT")
          set_property(GLOBAL APPEND
            PROPERTY
              "_svtk_module_test_modules" "${_svtk_scan_module}")
          set(_svtk_scan_test_wants _svtk_scan_wants_marker ${${_svtk_scan_module}_TEST_DEPENDS})
        elseif (_svtk_scan_ENABLE_TESTS STREQUAL "DEFAULT")
          set_property(GLOBAL APPEND
            PROPERTY
              "_svtk_module_test_modules" "${_svtk_scan_module}")
        elseif (_svtk_scan_ENABLE_TESTS STREQUAL "OFF")
          # Nothing to do.
        else ()
          message(FATAL_ERROR
            "Unrecognized option for ENABLE_TESTS: ${_svtk_module_ENABLE_TESTS}.")
        endif ()
      endif ()

      # Add all dependent modules to the list of required or provided modules.
      set(_svtk_scan_is_wanting 0)
      foreach (_svtk_scan_module_depend IN LISTS "${_svtk_scan_module}_DEPENDS" "${_svtk_scan_module}_PRIVATE_DEPENDS" _svtk_scan_test_depends _svtk_scan_test_wants)
        if (_svtk_scan_module_depend STREQUAL "_svtk_scan_wants_marker")
          set(_svtk_scan_is_wanting 1)
          continue ()
        endif ()
        # Though we need to error if this would cause a disabled module to be
        # provided.
        if (_svtk_scan_module_depend IN_LIST _svtk_scan_disabled_modules)
          if (_svtk_scan_is_wanting)
            continue ()
          else ()
            message(FATAL_ERROR
              "The ${_svtk_scan_module} module requires the disabled module ${_svtk_scan_module_depend}.")
          endif ()
        endif ()

        if (DEFINED "_svtk_scan_provide_${_svtk_scan_module_depend}")
          if (NOT _svtk_scan_provide_${_svtk_scan_module_depend})
            message(FATAL_ERROR
              "The `${_svtk_scan_module_depend} should be provided, but is disabled.")
          endif ()
          continue ()
        endif ()
        set("_svtk_scan_provide_${_svtk_scan_module_depend}" ON)

        if (NOT _svtk_scan_module_depend IN_LIST _svtk_scan_current_modules)
          if (NOT TARGET "${_svtk_scan_module_depend}")
            _svtk_module_debug(provide "@_svtk_scan_module_depend@ is external and required due to dependency from @_svtk_scan_module@")
          endif ()
          list(APPEND _svtk_scan_required_modules
            "${_svtk_scan_module_depend}")
        else ()
          _svtk_module_debug(provide "@_svtk_scan_module_depend@ is provided due to dependency from @_svtk_scan_module@")
          list(APPEND _svtk_scan_provided_modules
            "${_svtk_scan_module_depend}")
        endif ()
      endforeach ()
    endif ()
  endforeach ()

  if (_svtk_scan_provided_modules)
    list(REMOVE_DUPLICATES _svtk_scan_provided_modules)
  endif ()

  set(_svtk_scan_provided_kits)

  # Build a list of kits which contain the provided modules.
  foreach (_svtk_scan_provided_module IN LISTS _svtk_scan_provided_modules)
    if (${_svtk_scan_provided_module}_KIT)
      list(APPEND _svtk_scan_provided_kits
        "${${_svtk_scan_provided_module}_KIT}")
      set_property(GLOBAL APPEND
        PROPERTY
          "_svtk_kit_${${_svtk_scan_provided_module}_KIT}_kit_modules" "${_svtk_scan_provided_module}")
    endif ()
  endforeach ()

  if (_svtk_scan_provided_kits)
    list(REMOVE_DUPLICATES _svtk_scan_provided_kits)
  endif ()

  if (_svtk_scan_required_modules)
    list(REMOVE_DUPLICATES _svtk_scan_required_modules)
  endif ()

  set(_svtk_scan_unrecognized_modules
    ${_svtk_scan_REQUEST_MODULES}
    ${_svtk_scan_REJECT_MODULES})

  if (_svtk_scan_unrecognized_modules AND (_svtk_scan_provided_modules OR _svtk_scan_rejected_modules))
    list(REMOVE_ITEM _svtk_scan_unrecognized_modules
      ${_svtk_scan_provided_modules}
      ${_svtk_scan_rejected_modules})
  endif ()

  set("${_svtk_scan_PROVIDES_MODULES}"
    ${_svtk_scan_provided_modules}
    PARENT_SCOPE)

  if (DEFINED _svtk_scan_REQUIRES_MODULES)
    set("${_svtk_scan_REQUIRES_MODULES}"
      ${_svtk_scan_required_modules}
      PARENT_SCOPE)
  endif ()

  if (DEFINED _svtk_scan_UNRECOGNIZED_MODULES)
    set("${_svtk_scan_UNRECOGNIZED_MODULES}"
      ${_svtk_scan_unrecognized_modules}
      PARENT_SCOPE)
  endif ()

  if (DEFINED _svtk_scan_PROVIDES_KITS)
    set("${_svtk_scan_PROVIDES_KITS}"
      ${_svtk_scan_provided_kits}
      PARENT_SCOPE)
  endif ()
endfunction ()

#[==[
@page module-overview

@section module-target-functions Module-as-target functions

Due to the nature of SVTK modules supporting being built as kits, the module
name might not be usable as a target to CMake's `target_` family of commands.
Instead, there are various wrappers around them which take the module name as
an argument. These handle the forwarding of relevant information to the kit
library as well where necessary.

  - @ref svtk_module_set_properties
  - @ref svtk_module_set_property
  - @ref svtk_module_get_property
  - @ref svtk_module_depend
  - @ref svtk_module_include
  - @ref svtk_module_definitions
  - @ref svtk_module_compile_options
  - @ref svtk_module_compile_features
  - @ref svtk_module_link
  - @ref svtk_module_link_options
#]==]

#[==[
@page module-internal-api

@section module-target-internals Module target internals

When manipulating modules as targets, there are a few functions provided for
use in wrapping code to more easily access them.

  - @ref _svtk_module_real_target
  - @ref _svtk_module_real_target_kit
#]==]

#[==[
@ingroup module-internal
@brief The real target for a module

~~~
_svtk_module_real_target(<var> <module>)
~~~

Sometimes the actual, core target for a module is required (e.g., setting
CMake-level target properties or install rules). This function returns the real
target for a module.
#]==]
function (_svtk_module_real_target var module)
  if (ARGN)
    message(FATAL_ERROR
      "Unparsed arguments for _svtk_module_real_target: ${ARGN}.")
  endif ()

  set(_svtk_real_target_res "")
  if (TARGET "${module}")
    get_property(_svtk_real_target_imported
      TARGET    "${module}"
      PROPERTY  IMPORTED)
    if (_svtk_real_target_imported)
      set(_svtk_real_target_res "${module}")
    endif ()
  endif ()

  if (NOT _svtk_real_target_res)
    get_property(_svtk_real_target_res GLOBAL
      PROPERTY "_svtk_module_${module}_target_name")
    # Querying during the build.
    if (DEFINED _svtk_build_BUILD_WITH_KITS AND _svtk_build_BUILD_WITH_KITS)
      get_property(_svtk_real_target_kit GLOBAL
        PROPERTY "_svtk_module_${module}_kit")
      if (_svtk_real_target_kit)
        set(_svtk_real_target_res "${_svtk_real_target_res}-objects")
      endif ()
    # A query for after the module is built.
    elseif (TARGET "${_svtk_real_target_res}-objects")
      set(_svtk_real_target_res "${_svtk_real_target_res}-objects")
    endif ()
  endif ()

  if (NOT _svtk_real_target_res)
    set(_svtk_real_target_msg "")
    if (NOT TARGET "${module}")
      if (DEFINED _svtk_build_module)
        set(_svtk_real_target_msg
          " Is a module dependency missing?")
      else ()
        set(_svtk_real_target_msg
          " Is a `find_package` missing a required component?")
      endif ()
    endif ()
    message(FATAL_ERROR
      "Failed to determine the real target for the `${module}` "
      "module.${_svtk_real_target_msg}")
  endif ()

  set("${var}"
    "${_svtk_real_target_res}"
    PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module-internal
@brief The real target for a kit

~~~
_svtk_module_real_target_kit(<var> <kit>)
~~~

Sometimes the actual, core target for a module is required (e.g., setting
CMake-level target properties or install rules). This function returns the real
target for a kit.
#]==]
function (_svtk_module_real_target_kit var kit)
  if (ARGN)
    message(FATAL_ERROR
      "Unparsed arguments for _svtk_module_real_target_kit: ${ARGN}.")
  endif ()

  set(_svtk_real_target_res "")
  if (TARGET "${kit}")
    get_property(_svtk_real_target_imported
      TARGET    "${kit}"
      PROPERTY  IMPORTED)
    if (_svtk_real_target_imported)
      set(_svtk_real_target_res "${kit}")
    endif ()
  endif ()

  if (NOT _svtk_real_target_res)
    get_property(_svtk_real_target_res GLOBAL
      PROPERTY "_svtk_kit_${kit}_target_name")
  endif ()

  if (NOT _svtk_real_target_res)
    message(FATAL_ERROR
      "Failed to determine the real target for the `${kit}` kit.")
  endif ()

  set("${var}"
    "${_svtk_real_target_res}"
    PARENT_SCOPE)
endfunction ()

#[==[
@ingroup module
@brief Set multiple properties on a module

A wrapper around `set_target_properties` that works for modules.

~~~
svtk_module_set_properties(<module>
  [<property> <value>]...)
~~~
#]==]
function (svtk_module_set_properties module)
  _svtk_module_real_target(_svtk_set_properties_target "${module}")

  set_target_properties("${_svtk_set_properties_target}"
    PROPERTIES
      ${ARGN})
endfunction ()

#[==[
@ingroup module
@brief Set a property on a module

A wrapper around `set_property(TARGET)` that works for modules.

~~~
svtk_module_set_property(<module>
  [APPEND] [APPEND_STRING]
  PROPERTY  <property>
  VALUE     <value>...)
~~~
#]==]
function (svtk_module_set_property module)
  cmake_parse_arguments(_svtk_property
    "APPEND;APPEND_STRING"
    "PROPERTY"
    "VALUE"
    ${ARGN})

  if (_svtk_property_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_set_property: "
      "${_svtk_property_UNPARSED_ARGUMENTS}.")
  endif ()

  if (NOT DEFINED _svtk_property_PROPERTY)
    message(FATAL_ERROR
      "The `PROPERTY` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_property_VALUE)
    message(FATAL_ERROR
      "The `VALUE` argument is required.")
  endif ()

  if (_svtk_property_APPEND AND _svtk_property_APPEND_STRING)
    message(FATAL_ERROR
      "`APPEND` and `APPEND_STRING` may not be used at the same time.")
  endif ()

  set(_svtk_property_args)
  if (_svtk_property_APPEND)
    list(APPEND _svtk_property_args
      APPEND)
  endif ()
  if (_svtk_property_APPEND_STRING)
    list(APPEND _svtk_property_args
      APPEND_STRING)
  endif ()

  _svtk_module_real_target(_svtk_property_target "${module}")

  set_property(TARGET "${_svtk_property_target}"
    ${_svtk_property_args}
    PROPERTY
      "${_svtk_property_PROPERTY}" "${_svtk_property_VALUE}")
endfunction ()

#[==[
@ingroup module
@brief Get a property from a module

A wrapper around `get_property(TARGET)` that works for modules.

~~~
svtk_module_get_property(<module>
  PROPERTY  <property>
  VARIABLE  <variable>)
~~~

The variable name passed to the `VARIABLE` argument will be unset if the
property is not set (rather than the empty string).
#]==]
function (svtk_module_get_property module)
  cmake_parse_arguments(_svtk_property
    ""
    "PROPERTY;VARIABLE"
    ""
    ${ARGN})

  if (_svtk_property_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_get_property: "
      "${_svtk_property_UNPARSED_ARGUMENTS}.")
  endif ()

  if (NOT DEFINED _svtk_property_PROPERTY)
    message(FATAL_ERROR
      "The `PROPERTY` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_property_VARIABLE)
    message(FATAL_ERROR
      "The `VARIABLE` argument is required.")
  endif ()

  _svtk_module_real_target(_svtk_property_target "${module}")

  get_property(_svtk_property_is_set
    TARGET    "${_svtk_property_target}"
    PROPERTY  "${_svtk_property_PROPERTY}"
    SET)
  if (_svtk_property_is_set)
    get_property(_svtk_property_value
      TARGET    "${_svtk_property_target}"
      PROPERTY  "${_svtk_property_PROPERTY}")

    set("${_svtk_property_VARIABLE}"
      "${_svtk_property_value}"
      PARENT_SCOPE)
  else ()
    unset("${_svtk_property_VARIABLE}"
      PARENT_SCOPE)
  endif ()
endfunction ()

#[==[
@ingroup module-impl
@brief Generate arguments for target function wrappers

Create the `INTERFACE`, `PUBLIC`, and `PRIVATE` arguments for a function
wrapping CMake's `target_` functions to call the wrapped function.

This is necessary because not all of the functions support empty lists given a
keyword.
#]==]
function (_svtk_module_target_function prefix)
  foreach (visibility IN ITEMS INTERFACE PUBLIC PRIVATE)
    if (${prefix}_${visibility})
      set("${prefix}_${visibility}_args"
        "${visibility}"
        ${${prefix}_${visibility}}
        PARENT_SCOPE)
    endif ()
  endforeach ()
endfunction ()

#[==[
@ingroup module
@brief Add dependencies to a module

A wrapper around `add_dependencies` that works for modules.

~~~
svtk_module_depend(<module> <depend>...)
~~~
#]==]
function (svtk_module_depend module)
  _svtk_module_real_target(_svtk_depend_target "${module}")

  add_dependencies("${_svtk_depend_target}"
    ${ARGN})
endfunction ()

#[==[
@ingroup module
@brief Add include directories to a module

A wrapper around `add_dependencies` that works for modules.

~~~
svtk_module_include(<module>
  [SYSTEM]
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_include module)
  cmake_parse_arguments(_svtk_include
    "SYSTEM"
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_include_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_include: "
      "${_svtk_include_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_include_target "${module}")
  _svtk_module_target_function(_svtk_include)

  set(_svtk_include_system_arg)
  if (_svtk_include_SYSTEM)
    set(_svtk_include_system_arg SYSTEM)
  endif ()

  target_include_directories("${_svtk_include_target}"
    ${_svtk_include_system_arg}
    ${_svtk_include_INTERFACE_args}
    ${_svtk_include_PUBLIC_args}
    ${_svtk_include_PRIVATE_args})
endfunction ()

#[==[
@ingroup module
@brief Add compile definitions to a module

A wrapper around `target_compile_definitions` that works for modules.

~~~
svtk_module_definitions(<module>
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_definitions module)
  cmake_parse_arguments(_svtk_definitions
    ""
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_definitions_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_definitions: "
      "${_svtk_definitions_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_definitions_target "${module}")
  _svtk_module_target_function(_svtk_definitions)

  target_compile_definitions("${_svtk_definitions_target}"
    ${_svtk_definitions_INTERFACE_args}
    ${_svtk_definitions_PUBLIC_args}
    ${_svtk_definitions_PRIVATE_args})
endfunction ()

#[==[
@ingroup module
@brief Add compile options to a module

A wrapper around `target_compile_options` that works for modules.

~~~
svtk_module_compile_options(<module>
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_compile_options module)
  cmake_parse_arguments(_svtk_compile_options
    ""
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_compile_options_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_compile_options: "
      "${_svtk_compile_options_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_compile_options_target "${module}")
  _svtk_module_target_function(_svtk_compile_options)

  target_compile_options("${_svtk_compile_options_target}"
    ${_svtk_compile_options_INTERFACE_args}
    ${_svtk_compile_options_PUBLIC_args}
    ${_svtk_compile_options_PRIVATE_args})
endfunction ()

#[==[
@ingroup module
@brief Add compile features to a module

A wrapper around `target_compile_features` that works for modules.

~~~
svtk_module_compile_features(<module>
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_compile_features module)
  cmake_parse_arguments(_svtk_compile_features
    ""
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_compile_features_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_compile_features: "
      "${_svtk_compile_features_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_compile_features_target "${module}")
  _svtk_module_target_function(_svtk_compile_features)

  target_compile_features("${_svtk_compile_features_target}"
    ${_svtk_compile_features_INTERFACE_args}
    ${_svtk_compile_features_PUBLIC_args}
    ${_svtk_compile_features_PRIVATE_args})
endfunction ()

#[==[
@ingroup module
@brief Add link libraries to a module

A wrapper around `target_link_libraries` that works for modules. Note that this
function does extra work in kit builds, so circumventing it may break in kit
builds.

~~~
svtk_module_link(<module>
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_link module)
  cmake_parse_arguments(_svtk_link
    ""
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_link_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_link: "
      "${_svtk_link_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_link_target "${module}")
  _svtk_module_target_function(_svtk_link)

  get_property(_svtk_link_kit GLOBAL
    PROPERTY "_svtk_module_${module}_kit")
  if (_svtk_link_kit AND NOT CMAKE_VERSION VERSION_LESS "3.12")
    foreach (_svtk_link_private IN LISTS _svtk_link_PRIVATE)
      if (NOT TARGET "${_svtk_link_private}")
        continue ()
      endif ()

      get_property(_svtk_link_private_imported
        TARGET    "${_svtk_link_private}"
        PROPERTY  IMPORTED)
      if (_svtk_link_private_imported)
        get_property(_svtk_link_private_imported_global
          TARGET    "${_svtk_link_private}"
          PROPERTY  IMPORTED_GLOBAL)
        if (NOT _svtk_link_private_imported_global)
          set_property(TARGET "${_svtk_link_private}"
            PROPERTY
              IMPORTED_GLOBAL TRUE)
        endif ()
      endif ()
    endforeach ()
    set_property(GLOBAL APPEND
      PROPERTY
        "_svtk_kit_${_svtk_link_kit}_private_links" ${_svtk_link_PRIVATE})
  endif ()

  target_link_libraries("${_svtk_link_target}"
    ${_svtk_link_INTERFACE_args}
    ${_svtk_link_PUBLIC_args}
    ${_svtk_link_PRIVATE_args})
endfunction ()

#[==[
@ingroup module
@brief Add link options to a module

A wrapper around `target_link_options` that works for modules.

~~~
svtk_module_link_options(<module>
  [PUBLIC     <directory>...]
  [PRIVATE    <directory>...]
  [INTERFACE  <directory>...])
~~~
#]==]
function (svtk_module_link_options module)
  cmake_parse_arguments(_svtk_link_options
    ""
    ""
    "INTERFACE;PUBLIC;PRIVATE"
    ${ARGN})

  if (_svtk_link_options_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_link_options: "
      "${_svtk_link_options_UNPARSED_ARGUMENTS}.")
  endif ()

  _svtk_module_real_target(_svtk_link_options_target "${module}")
  _svtk_module_target_function(_svtk_link_options)

  target_link_options("${_svtk_link_options_target}"
    ${_svtk_link_options_INTERFACE_args}
    ${_svtk_link_options_PUBLIC_args}
    ${_svtk_link_options_PRIVATE_args})
endfunction ()

#[==[
@page module-internal-api

@ingroup module-internal
@section module-properties Module properties

The SVTK module system leverages CMake's target propagation and storage. As
such, there are a number of properties added to the targets representing
modules. These properties are intended for use by the module system and
associated functionality. In particular, more properties may be available by
language wrappers.

@subsection module-properties-naming Naming properties

When creating properties for use with the module system, they should be
prefixed with `INTERFACE_svtk_module_`. The `INTERFACE_` portion is required in
order to work with interface libraries. The `svtk_module_` portion is to avoid
colliding with any other properties. This function assumes this naming scheme
for some of its convenience features as well.

Properties should be the same in the local build as well as when imported to
ease use.

@subsection module-properties-system SVTK module system properties

There are a number of properties that are used and expected by the core of the
module system. These are generally module metadata (module dependencies,
whether to wrap or not, etc.). The properties all have the
`INTERFACE_svtk_module_` prefix mentioned in the previous section.

  * `third_party`: If set, the module represents a third party
    dependency and should be treated specially. Third party modules are very
    restricted and generally do not have any other properties set on them.
  * `exclude_wrap`: If set, the module should not be wrapped by an external
    language.
  * `depends`: The list of dependent modules. Language wrappers will generally
    require this to satisfy references to parent classes of the classes in the
    module.
  * `private_depends`: The list of privately dependent modules. Language
    wrappers may require this to satisfy references to parent classes of the
    classes in the module.
  * `optional_depends`: The list of optionally dependent modules. Language
    wrappers may require this to satisfy references to parent classes of the
    classes in the module.
  * `kit`: The kit the module is a member of. Only set if the module is
    actually a member of the kit (i.e., the module was built with
    `BUILD_WITH_KITS ON`).
  * `implements`: The list of modules for which this module registers to. This
    is used by the autoinit subsystem and generally is not required.
  * `implementable`: If set, this module provides registries which may be
    populated by dependent modules. It is used to check the `implements`
    property to help minimize unnecessary work from the autoinit subsystem.
  * `needs_autoinit`: If set, linking to this module requires the autoinit
    subsystem to ensure that registries in modules are fully populated.
  * `headers`: Paths to the public headers from the module. These are the
    headers which should be handled by language wrappers.
  * `hierarchy`: The path to the hierarchy file describing inheritance of the
    classes for use in language wrappers.
  * `forward_link`: Usage requirements that must be forwarded even though the
    module is linked to privately.

Kits have the following properties available (but only if kits are enabled):

  * `kit_modules`: Modules which are compiled into the kit.
#]==]

#[==[
@ingroup module-internal
@brief Set a module property

This function sets a [module property](@ref module-properties) on a module. The
required prefix will automatically be added to the passed name.

~~~
_svtk_module_set_module_property(<module>
  [APPEND] [APPEND_STRING]
  PROPERTY  <property>
  VALUE     <value>...)
~~~
#]==]
function (_svtk_module_set_module_property module)
  cmake_parse_arguments(_svtk_property
    "APPEND;APPEND_STRING"
    "PROPERTY"
    "VALUE"
    ${ARGN})

  if (_svtk_property_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_set_module_property: "
      "${_svtk_property_UNPARSED_ARGUMENTS}.")
  endif ()

  if (NOT DEFINED _svtk_property_PROPERTY)
    message(FATAL_ERROR
      "The `PROPERTY` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_property_VALUE)
    message(FATAL_ERROR
      "The `VALUE` argument is required.")
  endif ()

  if (_svtk_property_APPEND AND _svtk_property_APPEND_STRING)
    message(FATAL_ERROR
      "`APPEND` and `APPEND_STRING` may not be used at the same time.")
  endif ()

  set(_svtk_property_args)
  if (_svtk_property_APPEND)
    list(APPEND _svtk_property_args
      APPEND)
  endif ()
  if (_svtk_property_APPEND_STRING)
    list(APPEND _svtk_property_args
      APPEND_STRING)
  endif ()

  get_property(_svtk_property_is_alias
    TARGET    "${module}"
    PROPERTY  ALIASED_TARGET
    SET)
  if (_svtk_property_is_alias)
    _svtk_module_real_target(_svtk_property_target "${module}")
  else ()
    set(_svtk_property_target "${module}")
  endif ()

  set_property(TARGET "${_svtk_property_target}"
    ${_svtk_property_args}
    PROPERTY
      "INTERFACE_svtk_module_${_svtk_property_PROPERTY}" "${_svtk_property_VALUE}")
endfunction ()

#[==[
@ingroup module-internal
@brief Get a module property

Get a [module property](@ref module-properties) from a module.

~~~
_svtk_module_get_module_property(<module>
  PROPERTY  <property>
  VARIABLE  <variable>)
~~~

As with @ref svtk_module_get_property, the output variable will be unset if the
property is not set. The property name is automatically prepended with the
required prefix.
#]==]
function (_svtk_module_get_module_property module)
  cmake_parse_arguments(_svtk_property
    ""
    "PROPERTY;VARIABLE"
    ""
    ${ARGN})

  if (_svtk_property_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_get_module_property: "
      "${_svtk_property_UNPARSED_ARGUMENTS}.")
  endif ()

  if (NOT DEFINED _svtk_property_PROPERTY)
    message(FATAL_ERROR
      "The `PROPERTY` argument is required.")
  endif ()

  if (NOT DEFINED _svtk_property_VARIABLE)
    message(FATAL_ERROR
      "The `VARIABLE` argument is required.")
  endif ()

  get_property(_svtk_property_is_alias
    TARGET    "${module}"
    PROPERTY  ALIASED_TARGET
    SET)
  if (_svtk_property_is_alias)
    _svtk_module_real_target(_svtk_property_target "${module}")
  else ()
    set(_svtk_property_target "${module}")
  endif ()

  get_property(_svtk_property_is_set
    TARGET    "${_svtk_property_target}"
    PROPERTY  "INTERFACE_svtk_module_${_svtk_property_PROPERTY}"
    SET)
  if (_svtk_property_is_set)
    get_property(_svtk_property_value
      TARGET    "${_svtk_property_target}"
      PROPERTY  "INTERFACE_svtk_module_${_svtk_property_PROPERTY}")

    set("${_svtk_property_VARIABLE}"
      "${_svtk_property_value}"
      PARENT_SCOPE)
  else ()
    unset("${_svtk_property_VARIABLE}"
      PARENT_SCOPE)
  endif ()
endfunction ()

#[==[
@ingroup module-internal
@brief Check that destinations are valid

All installation destinations are expected to be relative so that
`CMAKE_INSTALL_PREFIX` can be relied upon in all code paths. This function may
be used to verify that destinations are relative.

~~~
_svtk_module_check_destinations(<prefix> [<suffix>...])
~~~

For each `suffix`, `prefix` is prefixed to it and the resulting variable name
is checked for validity as an install prefix. Raises an error if any is
invalid.
#]==]
function (_svtk_module_check_destinations prefix)
  foreach (suffix IN LISTS ARGN)
    if (IS_ABSOLUTE "${${prefix}${suffix}}")
      message(FATAL_ERROR
        "The `${suffix}` must not be an absolute path. Use "
        "`CMAKE_INSTALL_PREFIX` to keep everything in a single installation "
        "prefix.")
    endif ()
  endforeach ()
endfunction ()

#[==[
@ingroup module-internal
@brief Write an import prefix statement

CMake files, once installed, may need to construct paths to other locations
within the install prefix. This function writes a prefix computation for file
given its install destination.

~~~
_svtk_module_write_import_prefix(<file> <destination>)
~~~

The passed file is cleared so that it occurs at the top of the file. The prefix
is available in the file as the `_svtk_module_import_prefix` variable. It is
recommended to unset the variable at the end of the file.
#]==]
function (_svtk_module_write_import_prefix file destination)
  if (IS_ABSOLUTE "${destination}")
    message(FATAL_ERROR
      "An import prefix cannot be determined from an absolute installation "
      "destination. Use `CMAKE_INSTALL_PREFIX` to keep everything in a single "
      "installation prefix.")
  endif ()

  file(WRITE "${file}"
    "set(_svtk_module_import_prefix \"\${CMAKE_CURRENT_LIST_DIR}\")\n")
  while (destination)
    get_filename_component(destination "${destination}" DIRECTORY)
    file(APPEND "${file}"
      "get_filename_component(_svtk_module_import_prefix \"\${_svtk_module_import_prefix}\" DIRECTORY)\n")
  endwhile ()
endfunction ()

#[==[
@ingroup module-internal
@brief Export properties on modules and targets

This function is intended for use in support functions which leverage the
module system, not by general system users. This function supports exporting
properties from the build into dependencies via target properties which are
loaded from a project's config file which is loaded via CMake's `find_package`
function.

~~~
_svtk_module_export_properties(
  [MODULE       <module>]
  [KIT          <kit>]
  BUILD_FILE    <path>
  INSTALL_FILE  <path>
  [PROPERTIES               <property>...]
  [FROM_GLOBAL_PROPERTIES   <property fragment>...]
  [SPLIT_INSTALL_PROPERTIES <property fragment>...])
~~~

The `BUILD_FILE` and `INSTALL_FILE` arguments are required. Exactly one of
`MODULE` and `KIT` is also required. The `MODULE` or `KIT` argument holds the
name of the module or kit that will have properties exported. The `BUILD_FILE`
and `INSTALL_FILE` paths are *appended to*. As such, when setting up these
files, it should be preceded with:

~~~{.cmake}
file(WRITE "${build_file}")
file(WRITE "${install_file}")
~~~

To avoid accidental usage of the install file from the build tree, it is
recommended to store it under a `CMakeFiles/` directory in the build tree with
an additional `.install` suffix and use `install(RENAME)` to rename it at
install time.

The set of properties exported is computed as follows:

  * `PROPERTIES` queries the module target for the given property and exports
    its value as-is to both the build and install files. In addition, these
    properties are set on the target directly as the same name.
  * `FROM_GLOBAL_PROPERTIES` queries the global
    `_svtk_module_<MODULE>_<fragment>` property and exports it to both the build
    and install files as `INTERFACE_svtk_module_<fragment>`.
  * `SPLIT_INSTALL_PROPERTIES` queries the target for
    `INTERFACE_svtk_module_<fragment>` and exports its value to the build file
    and `INTERFACE_svtk_module_<fragment>_install` to the install file as the
    non-install property name. This is generally useful for properties which
    change between the build and installation.
#]==]
function (_svtk_module_export_properties)
  cmake_parse_arguments(_svtk_export_properties
    ""
    "BUILD_FILE;INSTALL_FILE;MODULE;KIT"
    "FROM_GLOBAL_PROPERTIES;PROPERTIES;SPLIT_INSTALL_PROPERTIES"
    ${ARGN})

  if (_svtk_export_properties_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for _svtk_export_properties: "
      "${_svtk_export_properties_UNPARSED_ARGUMENTS}.")
  endif ()

  if (DEFINED _svtk_export_properties_MODULE)
    if (DEFINED _svtk_export_properties_KIT)
      message(FATAL_ERROR
        "Only one of `MODULE` or `KIT` is required to export properties.")
    endif ()
    set(_svtk_export_properties_type "module")
    set(_svtk_export_properties_name "${_svtk_export_properties_MODULE}")
  elseif (_svtk_export_properties_KIT)
    set(_svtk_export_properties_type "kit")
    set(_svtk_export_properties_name "${_svtk_export_properties_KIT}")
  else ()
    message(FATAL_ERROR
      "A module or kit is required to export properties.")
  endif ()

  if (NOT _svtk_export_properties_BUILD_FILE)
    message(FATAL_ERROR
      "Exporting properties requires a build file to write to.")
  endif ()

  if (NOT _svtk_export_properties_INSTALL_FILE)
    message(FATAL_ERROR
      "Exporting properties requires an install file to write to.")
  endif ()

  if (_svtk_export_properties_type STREQUAL "module")
    _svtk_module_real_target(_svtk_export_properties_target_name "${_svtk_export_properties_name}")
  elseif (_svtk_export_properties_type STREQUAL "kit")
    _svtk_module_real_target_kit(_svtk_export_properties_target_name "${_svtk_export_properties_name}")
  endif ()

  foreach (_svtk_export_properties_global IN LISTS _svtk_export_properties_FROM_GLOBAL_PROPERTIES)
    get_property(_svtk_export_properties_is_set GLOBAL
      PROPERTY  "_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_name}_${_svtk_export_properties_global}"
      SET)
    if (NOT _svtk_export_properties_is_set)
      continue ()
    endif ()

    get_property(_svtk_export_properties_value GLOBAL
      PROPERTY  "_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_name}_${_svtk_export_properties_global}")
    set(_svtk_export_properties_set_property
      "set_property(TARGET \"${_svtk_export_properties_name}\" PROPERTY \"INTERFACE_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_global}\" \"${_svtk_export_properties_value}\")\n")

    set_property(TARGET "${_svtk_export_properties_target_name}"
      PROPERTY
        "INTERFACE_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_global}" "${_svtk_export_properties_value}")
    file(APPEND "${_svtk_export_properties_BUILD_FILE}"
      "${_svtk_export_properties_set_property}")
    file(APPEND "${_svtk_export_properties_INSTALL_FILE}"
      "${_svtk_export_properties_set_property}")
  endforeach ()

  foreach (_svtk_export_properties_target IN LISTS _svtk_export_properties_PROPERTIES)
    get_property(_svtk_export_properties_is_set
      TARGET    "${_svtk_export_properties_target_name}"
      PROPERTY  "${_svtk_export_properties_target}"
      SET)
    if (NOT _svtk_export_properties_is_set)
      continue ()
    endif ()

    get_property(_svtk_export_properties_value
      TARGET    "${_svtk_export_properties_target_name}"
      PROPERTY  "${_svtk_export_properties_target}")
    set(_svtk_export_properties_set_property
      "set_property(TARGET \"${_svtk_export_properties_name}\" PROPERTY \"${_svtk_export_properties_target}\" \"${_svtk_export_properties_value}\")\n")

    file(APPEND "${_svtk_export_properties_BUILD_FILE}"
      "${_svtk_export_properties_set_property}")
    file(APPEND "${_svtk_export_properties_INSTALL_FILE}"
      "${_svtk_export_properties_set_property}")
  endforeach ()

  foreach (_svtk_export_properties_split IN LISTS _svtk_export_properties_SPLIT_INSTALL_PROPERTIES)
    get_property(_svtk_export_properties_is_set
      TARGET    "${_svtk_export_properties_target_name}"
      PROPERTY  "INTERFACE_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_split}"
      SET)
    if (NOT _svtk_export_properties_is_set)
      continue ()
    endif ()

    get_property(_svtk_export_properties_value
      TARGET    "${_svtk_export_properties_target_name}"
      PROPERTY  "INTERFACE_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_split}")
    set(_svtk_export_properties_set_property
      "set_property(TARGET \"${_svtk_export_properties_name}\" PROPERTY \"INTERFACE_svtk_module_${_svtk_export_properties_split}\" \"${_svtk_export_properties_value}\")\n")
    file(APPEND "${_svtk_export_properties_BUILD_FILE}"
      "${_svtk_export_properties_set_property}")

    get_property(_svtk_export_properties_value
      TARGET    "${_svtk_export_properties_target_name}"
      PROPERTY  "INTERFACE_svtk_${_svtk_export_properties_type}_${_svtk_export_properties_split}_install")
    set(_svtk_export_properties_set_property
      "set_property(TARGET \"${_svtk_export_properties_name}\" PROPERTY \"INTERFACE_svtk_module_${_svtk_export_properties_split}\" \"${_svtk_export_properties_value}\")\n")
    file(APPEND "${_svtk_export_properties_INSTALL_FILE}"
      "${_svtk_export_properties_set_property}")
  endforeach ()
endfunction ()

include("${CMAKE_CURRENT_LIST_DIR}/svtkModuleTesting.cmake")

#[==[
@ingroup module
@brief Build modules and kits

Once all of the modules have been scanned, they need to be built. Generally,
there will be just one build necessary for a set of scans, though they may be
built distinctly as well. If there are multiple calls to this function, they
should generally in reverse order of their scans.

~~~
svtk_module_build(
  MODULES       <module>...
  [KITS          <kit>...]

  [LIBRARY_NAME_SUFFIX  <suffix>]
  [VERSION              <version>]
  [SOVERSION            <soversion>]

  [PACKAGE              <package>]

  [BUILD_WITH_KITS  <ON|OFF>]

  [ENABLE_WRAPPING <ON|OFF>]

  [USE_EXTERNAL <ON|OFF>]

  [INSTALL_HEADERS    <ON|OFF>]
  [HEADERS_COMPONENT  <component>]

  [TARGETS_COMPONENT  <component>]
  [INSTALL_EXPORT     <export>]

  [TEST_DIRECTORY_NAME        <name>]
  [TEST_DATA_TARGET           <target>]
  [TEST_INPUT_DATA_DIRECTORY  <directory>]
  [TEST_OUTPUT_DATA_DIRECTORY <directory>]
  [TEST_OUTPUT_DIRECTORY      <directory>]

  [ARCHIVE_DESTINATION    <destination>]
  [HEADERS_DESTINATION    <destination>]
  [LIBRARY_DESTINATION    <destination>]
  [RUNTIME_DESTINATION    <destination>]
  [CMAKE_DESTINATION      <destination>]
  [LICENSE_DESTINATION    <destination>]
  [HIERARCHY_DESTINATION  <destination>])
~~~

The only requirement of the function is the list of modules to build, the rest
have reasonable defaults if not specified.

  * `MODULES`: (Required) The list of modules to build.
  * `KITS`: (Required if `BUILD_WITH_KITS` is `ON`) The list of kits to build.
  * `LIBRARY_NAME_SUFFIX`: (Defaults to `""`) A suffix to add to library names.
    If it is not empty, it is prefixed with `-` to separate it from the kit
    name.
  * `VERSION`: If specified, the `VERSION` property on built libraries will be
    set to this value.
  * `SOVERSION`: If specified, the `SOVERSION` property on built libraries will
    be set to this value.
  * `PACKAGE`: (Defaults to `${CMAKE_PROJECT_NAME}`) The name the build is
    meant to be found as when using `find_package`. Note that separate builds
    will require distinct `PACKAGE` values.
  * `BUILD_WITH_KITS`: (Defaults to `OFF`) If enabled, kit libraries will be
    built.
  * `ENABLE_WRAPPING`: (Default depends on the existence of
    `SVTK::WrapHierarchy` or `SVTKCompileTools::WrapHierarchy` targets) If
    enabled, wrapping will be available to the modules built in this call.
  * `USE_EXTERNAL`: (Defaults to `OFF`) Whether third party modules should find
    external copies rather than building their own copy.
  * `INSTALL_HEADERS`: (Defaults to `ON`) Whether or not to install public headers.
  * `HEADERS_COMPONENT`: (Defaults to `development`) The install component to
    use for header installation. Note that other SDK-related bits use the same
    component (e.g., CMake module files).
  * `TARGETS_COMPONENT`: `Defaults to `runtime`) The install component to use
    for the libraries built.
  * `TARGET_NAMESPACE`: `Defaults to `\<AUTO\>`) The namespace for installed
    targets. All targets must have the same namespace. If set to `\<AUTO\>`,
    the namespace will be detected automatically.
  * `INSTALL_EXPORT`: (Defaults to `""`) If non-empty, targets will be added to
    the given export. The export will also be installed as part of this build
    command.
  * `TEST_DIRECTORY_NAME`: (Defaults to `Testing`) The name of the testing
    directory to look for in each module. Set to `NONE` to disable automatic
    test management.
  * `TEST_DATA_TARGET`: (Defaults to `<PACKAGE>-data`) The target to add
    testing data download commands to.
  * `TEST_INPUT_DATA_DIRECTORY`: (Defaults to
    `${CMAKE_CURRENT_SOURCE_DIR}/Data`) The directory which will contain data
    for use by tests.
  * `TEST_OUTPUT_DATA_DIRECTORY`: (Defaults to
    `${CMAKE_CURRENT_BINARY_DIR}/Data`) The directory which will contain data
    for use by tests.
  * `TEST_OUTPUT_DIRECTORY`: (Defaults to
    `${CMAKE_BINARY_DIR}/<TEST_DIRECTORY_NAME>/Temporary`) The directory which
    tests may write any output files to.

The remaining arguments control where to install files related to the build.
See CMake documentation for the difference between `ARCHIVE`, `LIBRARY`, and
`RUNTIME`.

  * `ARCHIVE_DESTINATION`: (Defaults to `${CMAKE_INSTALL_LIBDIR}`) The install
    destination for archive files.
  * `HEADERS_DESTINATION`: (Defaults to `${CMAKE_INSTALL_INCLUDEDIR}`) The
    install destination for header files.
  * `LIBRARY_DESTINATION`: (Defaults to `${CMAKE_INSTALL_LIBDIR}`) The install
    destination for library files.
  * `RUNTIME_DESTINATION`: (Defaults to `${CMAKE_INSTALL_BINDIR}`) The install
    destination for runtime files.
  * `CMAKE_DESTINATION`: (Defaults to `<library destination>/cmake/<package>`)
    The install destination for CMake files.
  * `LICENSE_DESTINATION`: (Defaults to `${CMAKE_INSTALL_DATAROOTDIR}/licenses/${CMAKE_PROJECT_NAME}`)
    The install destination for license files (relevant for third party
    packages).
  * `HIERARCHY_DESTINATION`: (Defaults to `<library
    destination>/svtk/hierarchy/<PACKAGE>`) The install destination
    for hierarchy files (used for language wrapping).
#]==]
function (svtk_module_build)
  set(_svtk_build_install_arguments
    # Headers
    INSTALL_HEADERS
    HEADERS_COMPONENT

    # Targets
    INSTALL_EXPORT
    TARGETS_COMPONENT
    TARGET_NAMESPACE

    # Destinations
    ARCHIVE_DESTINATION
    HEADERS_DESTINATION
    LIBRARY_DESTINATION
    RUNTIME_DESTINATION
    CMAKE_DESTINATION
    LICENSE_DESTINATION
    HIERARCHY_DESTINATION)
  set(_svtk_build_test_arguments
    # Testing
    TEST_DIRECTORY_NAME
    TEST_DATA_TARGET
    TEST_INPUT_DATA_DIRECTORY
    TEST_OUTPUT_DATA_DIRECTORY
    TEST_OUTPUT_DIRECTORY)

  # TODO: Add an option to build statically? Currently, `BUILD_SHARED_LIBS` is
  # used.

  cmake_parse_arguments(_svtk_build
    ""
    "BUILD_WITH_KITS;USE_EXTERNAL;LIBRARY_NAME_SUFFIX;VERSION;SOVERSION;PACKAGE;ENABLE_WRAPPING;${_svtk_build_install_arguments};${_svtk_build_test_arguments}"
    "MODULES;KITS"
    ${ARGN})

  if (_svtk_build_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_build: "
      "${_svtk_build_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_build_USE_EXTERNAL)
    set(_svtk_build_USE_EXTERNAL OFF)
  endif ()

  if (NOT DEFINED _svtk_build_PACKAGE)
    set(_svtk_build_PACKAGE svtk)
  endif ()
  get_property(_svtk_build_package_exists GLOBAL
    PROPERTY  "_svtk_module_package_${_svtk_build_PACKAGE}"
    SET)
  if (_svtk_build_package_exists)
    message(FATAL_ERROR
      "A set of modules have already been built using the "
      "`${_svtk_build_PACKAGE}` package.")
  else ()
    set_property(GLOBAL
      PROPERTY
        "_svtk_module_package_${_svtk_build_PACKAGE}" "ON")
  endif ()

  if (NOT DEFINED _svtk_build_INSTALL_HEADERS)
    set(_svtk_build_INSTALL_HEADERS ON)
  endif ()

  if (NOT DEFINED _svtk_build_ENABLE_WRAPPING)
    if (TARGET "SVTKCompileTools::WrapHierarchy" OR
        TARGET "SVTK::WrapHierarchy")
      set(_svtk_build_ENABLE_WRAPPING ON)
    else ()
      set(_svtk_build_ENABLE_WRAPPING OFF)
    endif ()
  endif ()

  if (NOT DEFINED _svtk_build_TARGET_NAMESPACE)
    set(_svtk_build_TARGET_NAMESPACE "<AUTO>")
  endif ()

  if (NOT DEFINED _svtk_build_BUILD_WITH_KITS)
    set(_svtk_build_BUILD_WITH_KITS OFF)
  endif ()

  if (_svtk_build_BUILD_WITH_KITS AND CMAKE_VERSION VERSION_LESS "3.12")
    message(FATAL_ERROR
      "Building with kits enabled requires CMake 3.12 which introduced "
      "support for OBJECT libraries to have and utilize usage requirements.")
  endif ()

  if (_svtk_build_BUILD_WITH_KITS AND NOT DEFINED _svtk_build_KITS)
    message(FATAL_ERROR
      "Building with kits was requested, but no kits were specified.")
  endif ()

  if (NOT DEFINED _svtk_build_TEST_DIRECTORY_NAME)
    set(_svtk_build_TEST_DIRECTORY_NAME "Testing")
  endif ()

  if (NOT DEFINED _svtk_build_TEST_DATA_TARGET)
    set(_svtk_build_TEST_DATA_TARGET "${_svtk_build_PACKAGE}-data")
  endif ()

  if (NOT DEFINED _svtk_build_TEST_INPUT_DATA_DIRECTORY)
    set(_svtk_build_TEST_INPUT_DATA_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Data")
  endif ()

  if (NOT DEFINED _svtk_build_TEST_OUTPUT_DATA_DIRECTORY)
    set(_svtk_build_TEST_OUTPUT_DATA_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Data")
  endif ()

  if (NOT DEFINED _svtk_build_TEST_OUTPUT_DIRECTORY)
    set(_svtk_build_TEST_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_build_TEST_DIRECTORY_NAME}/Temporary")
  endif ()

  if (NOT DEFINED _svtk_build_HEADERS_COMPONENT)
    set(_svtk_build_HEADERS_COMPONENT "development")
  endif ()

  if (NOT DEFINED _svtk_build_ARCHIVE_DESTINATION)
    set(_svtk_build_ARCHIVE_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif ()

  if (NOT DEFINED _svtk_build_HEADERS_DESTINATION)
    set(_svtk_build_HEADERS_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  endif ()

  if (NOT DEFINED _svtk_build_LIBRARY_DESTINATION)
    set(_svtk_build_LIBRARY_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif ()

  if (NOT DEFINED _svtk_build_RUNTIME_DESTINATION)
    set(_svtk_build_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif ()

  if (NOT DEFINED _svtk_build_CMAKE_DESTINATION)
    set(_svtk_build_CMAKE_DESTINATION "${_svtk_build_LIBRARY_DESTINATION}/cmake/${_svtk_build_PACKAGE}")
  endif ()

  if (NOT DEFINED _svtk_build_LICENSE_DESTINATION)
    set(_svtk_build_LICENSE_DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/licenses/${CMAKE_PROJECT_NAME}")
  endif ()

  if (NOT DEFINED _svtk_build_HIERARCHY_DESTINATION)
    set(_svtk_build_HIERARCHY_DESTINATION "${_svtk_build_LIBRARY_DESTINATION}/svtk/hierarchy/${_svtk_build_PACKAGE}")
  endif ()

  if (NOT DEFINED _svtk_build_TARGETS_COMPONENT)
    set(_svtk_build_TARGETS_COMPONENT "runtime")
  endif ()

  if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_build_ARCHIVE_DESTINATION}")
  endif ()
  if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_build_LIBRARY_DESTINATION}")
  endif ()
  if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_build_RUNTIME_DESTINATION}")
  endif ()

  if (NOT _svtk_build_MODULES)
    message(FATAL_ERROR
      "No modules given to build.")
  endif ()

  _svtk_module_check_destinations(_svtk_build_
    ARCHIVE_DESTINATION
    HEADERS_DESTINATION
    RUNTIME_DESTINATION
    CMAKE_DESTINATION
    LICENSE_DESTINATION
    HIERARCHY_DESTINATION)

  foreach (_svtk_build_module IN LISTS _svtk_build_MODULES)
    get_property("_svtk_build_${_svtk_build_module}_depends" GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_depends")
    get_property("_svtk_build_${_svtk_build_module}_private_depends" GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_private_depends")
    get_property("_svtk_build_${_svtk_build_module}_optional_depends" GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_optional_depends")
    get_property("_svtk_build_${_svtk_build_module}_order_depends" GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_order_depends")
    set("_svtk_build_${_svtk_build_module}_all_depends"
      ${_svtk_build_${_svtk_build_module}_depends}
      ${_svtk_build_${_svtk_build_module}_private_depends}
      ${_svtk_build_${_svtk_build_module}_optional_depends}
      ${_svtk_build_${_svtk_build_module}_order_depends})
  endforeach ()

  set(_svtk_build_sorted_modules "${_svtk_build_MODULES}")
  svtk_topological_sort(_svtk_build_sorted_modules "_svtk_build_" "_all_depends")

  foreach (_svtk_build_module IN LISTS _svtk_build_sorted_modules)
    if (NOT _svtk_build_module IN_LIST _svtk_build_MODULES)
      continue ()
    endif ()

    if (TARGET "${_svtk_build_module}")
      get_property(_svtk_build_is_imported
        TARGET    "${_svtk_build_module}"
        PROPERTY  IMPORTED)

      # TODO: Is this right?
      if (NOT _svtk_build_is_imported)
        message(FATAL_ERROR
          "The ${_svtk_build_module} module has been requested to be built, but "
          "it is already built by this project.")
      endif ()

      continue ()
    endif ()

    foreach (_svtk_build_depend IN LISTS "_svtk_build_${_svtk_build_module}_depends" "_svtk_build_${_svtk_build_module}_private_depends")
      if (NOT TARGET "${_svtk_build_depend}")
        message(FATAL_ERROR
          "The ${_svtk_build_depend} dependency is missing for ${_svtk_build_module}.")
      endif ()
    endforeach ()

    get_property(_svtk_build_module_file GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_file")
    if (NOT _svtk_build_module_file)
      message(FATAL_ERROR
        "The requested ${_svtk_build_module} module is not a SVTK module.")
    endif ()

    _svtk_module_debug(building "@_svtk_build_module@ is being built")

    get_filename_component(_svtk_build_module_dir "${_svtk_build_module_file}" DIRECTORY)
    file(RELATIVE_PATH _svtk_build_module_subdir "${CMAKE_SOURCE_DIR}" "${_svtk_build_module_dir}")
    add_subdirectory(
      "${CMAKE_SOURCE_DIR}/${_svtk_build_module_subdir}"
      "${CMAKE_BINARY_DIR}/${_svtk_build_module_subdir}")

    if (NOT TARGET "${_svtk_build_module}")
      message(FATAL_ERROR
        "The ${_svtk_build_module} is being built, but a matching target was "
        "not created.")
    endif ()
  endforeach ()

  if (_svtk_build_BUILD_WITH_KITS)
    foreach (_svtk_build_kit IN LISTS _svtk_build_KITS)
      get_property(_svtk_build_target_name GLOBAL
        PROPERTY  "_svtk_kit_${_svtk_build_kit}_target_name")
      set(_svtk_kit_source_file
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/svtk_module_kit_${_svtk_build_target_name}.c")
      file(GENERATE
        OUTPUT  "${_svtk_kit_source_file}"
        CONTENT "void svtk_module_kit_${_svtk_build_target_name}() {}\n")
      add_library("${_svtk_build_target_name}"
        "${_svtk_kit_source_file}")
      get_property(_svtk_build_namespace GLOBAL
        PROPERTY  "_svtk_kit_${_svtk_build_kit}_namespace")
      if (_svtk_build_TARGET_NAMESPACE STREQUAL "<AUTO>")
        set(_svtk_build_TARGET_NAMESPACE "${_svtk_build_namespace}")
      endif ()
      if (NOT _svtk_build_namespace STREQUAL _svtk_build_TARGET_NAMESPACE)
        message(FATAL_ERROR
          "The `TARGET_NAMESPACE` (${_svtk_build_TARGET_NAMESPACE}) is not the "
          "same as the ${_svtk_build_kit} kit namespace "
          "(${_svtk_build_namespace}).")
      endif ()
      if (NOT _svtk_build_kit STREQUAL _svtk_build_target_name)
        add_library("${_svtk_build_kit}" ALIAS
          "${_svtk_build_target_name}")
      endif ()
      _svtk_module_apply_properties("${_svtk_build_target_name}")
      _svtk_module_install("${_svtk_build_target_name}")

      set(_svtk_build_kit_modules_object_libraries)
      set(_svtk_build_kit_modules_private_depends)

      get_property(_svtk_build_kit_modules GLOBAL
        PROPERTY  "_svtk_kit_${_svtk_build_kit}_kit_modules")
      foreach (_svtk_build_kit_module IN LISTS _svtk_build_kit_modules)
        get_property(_svtk_build_kit_module_target_name GLOBAL
          PROPERTY "_svtk_module_${_svtk_build_kit_module}_target_name")
        list(APPEND _svtk_build_kit_modules_object_libraries
          "${_svtk_build_kit_module_target_name}-objects")

        # Since there is no link step for modules, we need to copy the private
        # dependencies of the constituent modules into the kit so that their
        # private dependencies are actually linked.
        get_property(_svtk_build_kit_module_private_depends GLOBAL
          PROPERTY "_svtk_module_${_svtk_build_kit_module}_private_depends")
        # Also grab optional dependencies since they end up being private
        # links.
        get_property(_svtk_build_kit_module_optional_depends GLOBAL
          PROPERTY "_svtk_module_${_svtk_build_kit_module}_optional_depends")
        foreach (_svtk_build_kit_module_private_depend IN LISTS _svtk_build_kit_module_private_depends _svtk_build_kit_module_optional_depends)
          if (NOT TARGET "${_svtk_build_kit_module_private_depend}")
            continue ()
          endif ()

          # But we don't need to link to modules that are part of the kit we are
          # building.
          if (NOT _svtk_build_kit_module_private_depend IN_LIST _svtk_build_kit_modules)
            list(APPEND _svtk_build_kit_modules_private_depends
              "$<LINK_ONLY:${_svtk_build_kit_module_private_depend}>")
          endif ()
        endforeach ()
      endforeach ()

      get_property(_svtk_build_kit_private_links GLOBAL
        PROPERTY "_svtk_kit_${_svtk_build_kit}_private_links")

      if (_svtk_build_kit_modules_private_depends)
        list(REMOVE_DUPLICATES _svtk_build_kit_modules_private_depends)
      endif ()
      if (_svtk_build_kit_modules_private_links)
        list(REMOVE_DUPLICATES _svtk_build_kit_modules_private_links)
      endif ()

      target_link_libraries("${_svtk_build_target_name}"
        PRIVATE
          ${_svtk_build_kit_modules_object_libraries}
          ${_svtk_build_kit_modules_private_depends}
          ${_svtk_build_kit_private_links})
      get_property(_svtk_build_kit_library_name GLOBAL
        PROPERTY "_svtk_kit_${_svtk_build_kit}_library_name")
      if (_svtk_build_LIBRARY_NAME_SUFFIX)
        string(APPEND _svtk_build_kit_library_name "-${_svtk_build_LIBRARY_NAME_SUFFIX}")
      endif ()
      set_target_properties("${_svtk_build_target_name}"
        PROPERTIES
          OUTPUT_NAME "${_svtk_build_kit_library_name}")
    endforeach ()
  endif ()

  set(_svtk_build_properties_filename "${_svtk_build_PACKAGE}-svtk-module-properties.cmake")
  set(_svtk_build_properties_install_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_build_properties_filename}.install")
  set(_svtk_build_properties_build_file "${CMAKE_BINARY_DIR}/${_svtk_build_CMAKE_DESTINATION}/${_svtk_build_properties_filename}")

  file(WRITE "${_svtk_build_properties_build_file}")

  _svtk_module_write_import_prefix(
    "${_svtk_build_properties_install_file}"
    "${_svtk_build_CMAKE_DESTINATION}")

  foreach (_svtk_build_module IN LISTS _svtk_build_MODULES)
    get_property(_svtk_build_namespace GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_namespace")
    if (_svtk_build_TARGET_NAMESPACE STREQUAL "<AUTO>")
      set(_svtk_build_TARGET_NAMESPACE "${_svtk_build_namespace}")
    endif ()
    if (NOT _svtk_build_namespace STREQUAL _svtk_build_TARGET_NAMESPACE)
      message(FATAL_ERROR
        "The `TARGET_NAMESPACE` (${_svtk_build_TARGET_NAMESPACE}) is not the "
        "same as the ${_svtk_build_module} module namespace "
        "(${_svtk_build_namespace}).")
    endif ()

    get_property(_svtk_build_is_third_party
      TARGET    "${_svtk_build_module}"
      PROPERTY  "INTERFACE_svtk_module_third_party")
    if (_svtk_build_is_third_party)
      _svtk_module_export_properties(
        BUILD_FILE    "${_svtk_build_properties_build_file}"
        INSTALL_FILE  "${_svtk_build_properties_install_file}"
        MODULE        "${_svtk_build_module}"
        FROM_GLOBAL_PROPERTIES
          # Export the dependencies of a module.
          depends
          private_depends
          optional_depends
          # The library name of the module.
          library_name
        PROPERTIES
          # Export whether a module is third party or not.
          INTERFACE_svtk_module_third_party
          INTERFACE_svtk_module_exclude_wrap)
      continue ()
    endif ()

    set(_svtk_build_split_properties)
    get_property(_svtk_build_exclude_wrap
      TARGET    "${_svtk_build_module}"
      PROPERTY  "INTERFACE_svtk_module_${_svtk_build_module}_exclude_wrap")
    if (NOT _svtk_build_exclude_wrap)
      list(APPEND _svtk_build_split_properties
        headers)
      if (_svtk_build_ENABLE_WRAPPING)
        list(APPEND _svtk_build_split_properties
          hierarchy)
      endif ()
    endif ()

    set(_svtk_build_properties_kit_properties)
    if (_svtk_build_BUILD_WITH_KITS)
      list(APPEND _svtk_build_properties_kit_properties
        # Export the kit membership of a module.
        kit)
    endif ()

    _svtk_module_export_properties(
      BUILD_FILE    "${_svtk_build_properties_build_file}"
      INSTALL_FILE  "${_svtk_build_properties_install_file}"
      MODULE        "${_svtk_build_module}"
      FROM_GLOBAL_PROPERTIES
        # Export whether the module should be excluded from wrapping or not.
        exclude_wrap
        # Export the dependencies of a module.
        depends
        private_depends
        optional_depends
        # Export what modules are implemented by the module.
        implements
        # Export whether the module contains autoinit logic.
        implementable
        # The library name of the module.
        library_name
        ${_svtk_build_properties_kit_properties}
      PROPERTIES
        # Export whether the module needs autoinit logic handled.
        INTERFACE_svtk_module_needs_autoinit
        # Forward private usage requirements with global effects.
        INTERFACE_svtk_module_forward_link
      SPLIT_INSTALL_PROPERTIES
        # Set the properties which differ between build and install trees.
        ${_svtk_build_split_properties})
  endforeach ()

  if (_svtk_build_BUILD_WITH_KITS)
    foreach (_svtk_build_kit IN LISTS _svtk_build_KITS)
      _svtk_module_export_properties(
        BUILD_FILE    "${_svtk_build_properties_build_file}"
        INSTALL_FILE  "${_svtk_build_properties_install_file}"
        KIT           "${_svtk_build_kit}"
        FROM_GLOBAL_PROPERTIES
          # Export the list of modules in the kit.
          kit_modules)
    endforeach ()
  endif ()

  if (_svtk_build_INSTALL_EXPORT AND _svtk_build_INSTALL_HEADERS)
    set(_svtk_build_namespace)
    if (_svtk_build_TARGET_NAMESPACE)
      set(_svtk_build_namespace
        NAMESPACE "${_svtk_build_TARGET_NAMESPACE}::")
    endif ()

    export(
      EXPORT    "${_svtk_build_INSTALL_EXPORT}"
      ${_svtk_build_namespace}
      FILE      "${CMAKE_BINARY_DIR}/${_svtk_build_CMAKE_DESTINATION}/${_svtk_build_PACKAGE}-targets.cmake")
    install(
      EXPORT      "${_svtk_build_INSTALL_EXPORT}"
      DESTINATION "${_svtk_build_CMAKE_DESTINATION}"
      ${_svtk_build_namespace}
      FILE        "${_svtk_build_PACKAGE}-targets.cmake"
      COMPONENT   "${_svtk_build_HEADERS_COMPONENT}")

    if (_svtk_build_INSTALL_HEADERS)
      file(APPEND "${_svtk_build_properties_install_file}"
        "unset(_svtk_module_import_prefix)\n")

      install(
        FILES       "${_svtk_build_properties_install_file}"
        DESTINATION "${_svtk_build_CMAKE_DESTINATION}"
        RENAME      "${_svtk_build_properties_filename}"
        COMPONENT   "${_svtk_build_HEADERS_COMPONENT}")
    endif ()
  endif ()

  get_property(_svtk_build_test_modules GLOBAL
    PROPERTY "_svtk_module_test_modules")
  set(_svtk_build_tests_handled)
  foreach (_svtk_build_test IN LISTS _svtk_build_test_modules)
    if (NOT _svtk_build_test IN_LIST _svtk_build_MODULES)
      continue ()
    endif ()
    list(APPEND _svtk_build_tests_handled
      "${_svtk_build_test}")

    get_property(_svtk_build_test_depends GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_test}_test_depends")

    set(_svtk_build_test_has_depends TRUE)
    foreach (_svtk_build_test_depend IN LISTS _svtk_build_test_depends)
      if (NOT TARGET "${_svtk_build_test_depend}")
        set(_svtk_build_test_has_depends FALSE)
        _svtk_module_debug(testing "@_svtk_build_test@ testing disabled due to missing @_svtk_build_test_depend@")
      endif ()
    endforeach ()
    if (NOT _svtk_build_test_has_depends)
      continue ()
    endif ()

    get_property(_svtk_build_module_file GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_test}_file")

    if (NOT _svtk_build_TEST_DIRECTORY_NAME STREQUAL "NONE")
      get_filename_component(_svtk_build_module_dir "${_svtk_build_module_file}" DIRECTORY)
      file(RELATIVE_PATH _svtk_build_module_subdir "${CMAKE_SOURCE_DIR}" "${_svtk_build_module_dir}")
      if (EXISTS "${CMAKE_SOURCE_DIR}/${_svtk_build_module_subdir}/${_svtk_build_TEST_DIRECTORY_NAME}")
        get_property(_svtk_build_test_labels GLOBAL
          PROPERTY  "_svtk_module_${_svtk_build_test}_test_labels")
        add_subdirectory(
          "${CMAKE_SOURCE_DIR}/${_svtk_build_module_subdir}/${_svtk_build_TEST_DIRECTORY_NAME}"
          "${CMAKE_BINARY_DIR}/${_svtk_build_module_subdir}/${_svtk_build_TEST_DIRECTORY_NAME}")
      endif ()
    endif ()
  endforeach ()

  if (_svtk_build_test_modules AND _svtk_build_tests_handled)
    list(REMOVE_ITEM _svtk_build_test_modules
      ${_svtk_build_tests_handled})
    set_property(GLOBAL
      PROPERTY
        _svtk_module_test_modules "${_svtk_build_test_modules}")
  endif ()
endfunction ()

#[==[
@ingroup module-impl
@brief Add "standard" include directories to a module

Add the "standard" includes for a module to its interface. These are the source
and build directories for the module itself. They are always either `PUBLIC` or
`INTERFACE` (depending on the module's target type).

~~~
_svtk_module_standard_includes(
  [SYSTEM]
  [INTERFACE]
  TARGET                <target>
  [HEADERS_DESTINATION  <destination>])
~~~
#]==]
function (_svtk_module_standard_includes)
  cmake_parse_arguments(_svtk_standard_includes
    "SYSTEM;INTERFACE"
    "TARGET;HEADERS_DESTINATION"
    ""
    ${ARGN})

  if (NOT _svtk_standard_includes_TARGET)
    message(FATAL_ERROR
      "The `TARGET` argument is required.")
  endif ()
  if (NOT TARGET "${_svtk_standard_includes_TARGET}")
    message(FATAL_ERROR
      "The `TARGET` argument is not a target.")
  endif ()

  if (_svtk_standard_includes_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_standard_includes: "
      "${_svtk_standard_includes_UNPARSED_ARGUMENTS}")
  endif ()

  set(_svtk_standard_includes_system)
  if (_svtk_standard_includes_SYSTEM)
    set(_svtk_standard_includes_system SYSTEM)
  endif ()

  set(_svtk_standard_includes_visibility PUBLIC)
  if (_svtk_standard_includes_INTERFACE)
    set(_svtk_standard_includes_visibility INTERFACE)
  endif ()

  target_include_directories("${_svtk_standard_includes_TARGET}"
    ${_svtk_standard_includes_system}
    "${_svtk_standard_includes_visibility}"
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

  if (_svtk_build_INSTALL_HEADERS AND _svtk_standard_includes_HEADERS_DESTINATION)
    target_include_directories("${_svtk_standard_includes_TARGET}"
      ${_svtk_standard_includes_system}
      "${_svtk_standard_includes_visibility}"
      $<INSTALL_INTERFACE:${_svtk_standard_includes_HEADERS_DESTINATION}>)
  endif ()
endfunction ()

#[==[
@ingroup module-impl
@brief Determine the default export macro for a module

Determines the export macro to be used for a module from its metadata. Assumes
it is called from within a @ref svtk_module_build call.

~~~
_svtk_module_default_library_name(<varname>)
~~~
#]==]
function (_svtk_module_default_export_macro_prefix varname)
  get_property(_svtk_module_default_library_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  string(TOUPPER "${_svtk_module_default_library_name}" _svtk_default_export_macro_upper)
  set("${varname}"
    "${_svtk_default_export_macro_upper}"
    PARENT_SCOPE)
endfunction ()

# TODO: It would be nice to support `USE_LINK_PROPERTIES` instead of listing
# the modules again here. However, the format of the `LINK_LIBRARIES` property
# value may not be easy to handle.

#[==[
@page module-overview

@ingroup module
@section module-autoinit Autoinit

When a module contains a factory which may be populated by other modules, these
factories need to be populated when the modules are loaded by the dynamic linker
(for shared builds) or program load time (for static builds). To provide for
this, the module system contains an autoinit "subsystem".

@subsection module-autoinit-leverage Leveraging the autoinit subsystem

The subsystem provides the following hooks for use by projects:

  * In modules which `IMPLEMENTS` other modules, in the generated
    `<module>Module.h` header (which provides export symbols as well) will
    include the modules which are implemented.
  * In modules which are `IMPLEMENTABLE` or `IMPLEMENTS` another module, the
    generated `<module>Module.h` file will include the following block:

~~~{.c}
#ifdef <module>_AUTOINIT_INCLUDE
#include <module>_AUTOINIT_INCLUDE
#endif
#ifdef <module>_AUTOINIT
#include <header>
SVTK_MODULE_AUTOINIT(<module>)
#endif
~~~

The @ref svtk_module_autoinit function will generate an include file and provide
its path via the `<module>_AUTOINIT_INCLUDE` define. once it has been included,
if the `<module>_AUTOINIT` symbol is defined, a header is included which is
intended to provide the `SVTK_MODULE_AUTOINIT` macro. This macro is given the
module name and should use `<module>_AUTOINIT` to fill in the factories in the
module with those from the `IMPLEMENTS` modules listed in that symbol.

The `<module>_AUTOINIT` symbol's value is:

~~~
<count>(<module1>,<module2>,<module3>)
~~~

where `<count>` is the number of modules in the parentheses and each module
listed need to register something to `<module>`.

If not provided via the `AUTOINIT_INCLUDE` argument to the
@ref svtk_module_add_module function, the header to use is fetched from the
`_svtk_module_autoinit_include` global property. This only needs to be managed
in modules that `IMPLEMENTS` or are `IMPLEMENTABLE`. This should be provided by
projects using the module system at its lowest level. Projects not implementing
the `SVTK_MODULE_AUTOINIT` macro should have its value provided by
`find_package` dependencies in some way.
#]==]

#[==[
@ingroup module
@brief Linking to autoinit-using modules

When linking to modules, in order for the autoinit system to work, modules need
to declare their registration. In order to do this, defines may need to be
provided to targets in order to trigger registration. These defines may be
added to targets by using this function.

~~~
svtk_module_autoinit(
  TARGETS <target>...
  MODULES <module>...)
~~~

After this call, the targets given to the `TARGETS` argument will gain the
preprocessor definitions to trigger registrations properly.
#]==]
function (svtk_module_autoinit)
  cmake_parse_arguments(_svtk_autoinit
    ""
    ""
    "TARGETS;MODULES"
    ${ARGN})

  if (_svtk_autoinit_UNRECOGNIZED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_autoinit: "
      "${_svtk_autoinit_UNRECOGNIZED_ARGUMENTS}.")
  endif ()

  if (NOT _svtk_autoinit_TARGETS)
    message(FATAL_ERROR
      "The `TARGETS` argument is required.")
  endif ()

  if (NOT _svtk_autoinit_MODULES)
    message(AUTHOR_WARNING
      "No `MODULES` passed to `svtk_modules_autoinit`.")
  endif ()

  set(_svtk_autoinit_module_stack
    ${_svtk_autoinit_MODULES})

  set(_svtk_autoinit_needs_implements)
  set(_svtk_autoinit_seen)
  while (_svtk_autoinit_module_stack)
    list(GET _svtk_autoinit_module_stack 0 _svtk_autoinit_current_module)
    list(REMOVE_AT _svtk_autoinit_module_stack 0)
    if (_svtk_autoinit_current_module IN_LIST _svtk_autoinit_seen)
      continue ()
    endif ()
    list(APPEND _svtk_autoinit_seen
      "${_svtk_autoinit_current_module}")

    _svtk_module_real_target(_svtk_autoinit_current_target "${_svtk_autoinit_current_module}")
    get_property(_svtk_autoinit_implements
      TARGET    "${_svtk_autoinit_current_target}"
      PROPERTY  "INTERFACE_svtk_module_implements")

    list(APPEND _svtk_autoinit_needs_implements
      ${_svtk_autoinit_implements})
    foreach (_svtk_autoinit_implement IN LISTS _svtk_autoinit_implements)
      _svtk_module_real_target(_svtk_autoinit_implements_target "${_svtk_autoinit_implement}")
      get_property(_svtk_autoinit_implementable
        TARGET    "${_svtk_autoinit_implements_target}"
        PROPERTY  "INTERFACE_svtk_module_implementable")

      if (NOT _svtk_autoinit_implementable)
        message(FATAL_ERROR
          "The `${_svtk_autoinit_current_module}` module says that it "
          "implements the `${_svtk_autoinit_implement}` module, but it is not "
          "implementable.")
      endif ()

      list(APPEND "_svtk_autoinit_implements_${_svtk_autoinit_implement}"
        "${_svtk_autoinit_current_module}")
    endforeach ()
  endwhile ()

  if (NOT _svtk_autoinit_needs_implements)
    return ()
  endif ()
  list(REMOVE_DUPLICATES _svtk_autoinit_needs_implements)
  list(SORT _svtk_autoinit_needs_implements)

  set(_svtk_autoinit_hash_content)
  foreach (_svtk_autoinit_need_implements IN LISTS _svtk_autoinit_needs_implements)
    if (NOT _svtk_autoinit_implements_${_svtk_autoinit_need_implements})
      continue ()
    endif ()
    list(SORT "_svtk_autoinit_implements_${_svtk_autoinit_need_implements}")

    string(APPEND _svtk_autoinit_hash_content
      "${_svtk_autoinit_need_implements}: ${_svtk_autoinit_implements_${_svtk_autoinit_need_implements}}\n")
  endforeach ()
  string(MD5 _svtk_autoinit_header_tag "${_svtk_autoinit_hash_content}")
  set(_svtk_autoinit_header
    "${CMAKE_BINARY_DIR}/CMakeFiles/svtkModuleAutoInit_${_svtk_autoinit_header_tag}.h")

  get_property(_svtk_autoinit_header_generated GLOBAL
    PROPERTY "_svtk_autoinit_generated_${_svtk_autoinit_header_tag}")

  set(_svtk_autoinit_defines)
  set(_svtk_autoinit_header_content)
  foreach (_svtk_autoinit_need_implements IN LISTS _svtk_autoinit_needs_implements)
    if (NOT _svtk_autoinit_implements_${_svtk_autoinit_need_implements})
      continue ()
    endif ()

    get_property(_svtk_autoinit_implements_library_name
      TARGET    "${_svtk_autoinit_need_implements}"
      PROPERTY  "INTERFACE_svtk_module_library_name")

    if (NOT _svtk_autoinit_header_generated)
      list(LENGTH "_svtk_autoinit_implements_${_svtk_autoinit_need_implements}"
        _svtk_autoinit_length)
      set(_svtk_autoinit_args)
      foreach (_svtk_autoinit_arg IN LISTS "_svtk_autoinit_implements_${_svtk_autoinit_need_implements}")
        get_property(_svtk_autoinit_arg_library_name
          TARGET    "${_svtk_autoinit_arg}"
          PROPERTY  "INTERFACE_svtk_module_library_name")
        list(APPEND _svtk_autoinit_args
          "${_svtk_autoinit_arg_library_name}")
      endforeach ()
      string(REPLACE ";" "," _svtk_autoinit_args "${_svtk_autoinit_args}")
      string(APPEND _svtk_autoinit_header_content
        "#define ${_svtk_autoinit_implements_library_name}_AUTOINIT ${_svtk_autoinit_length}(${_svtk_autoinit_args})\n")
    endif ()

    list(APPEND _svtk_autoinit_defines
      "${_svtk_autoinit_implements_library_name}_AUTOINIT_INCLUDE=\"${_svtk_autoinit_header}\"")
  endforeach ()

  if (NOT _svtk_autoinit_header_generated)
    file(GENERATE
      OUTPUT  "${_svtk_autoinit_header}"
      CONTENT "${_svtk_autoinit_header_content}")

    set_property(GLOBAL
      PROPERTY
        "_svtk_autoinit_generated_${_svtk_autoinit_header_tag}" TRUE)
  endif ()

  foreach (_svtk_autoinit_target IN LISTS _svtk_autoinit_TARGETS)
    get_property(_svtk_autoinit_target_type
      TARGET    "${_svtk_autoinit_target}"
      PROPERTY  TYPE)
    if (_svtk_autoinit_target_type STREQUAL "INTERFACE_LIBRARY")
      continue ()
    endif ()

    target_compile_definitions("${_svtk_autoinit_target}"
      PRIVATE
        ${_svtk_autoinit_defines})
  endforeach ()
endfunction ()

#[==[
@ingroup module-impl
@brief Generate the hierarchy for a module

Write wrap hierarchy files for the module currently being built. This also
installs the hierarchy file for use by dependent projects if `INSTALL_HEADERS`
is set.

~~~
_svtk_module_write_wrap_hierarchy()
~~~
#]==]
function (_svtk_module_write_wrap_hierarchy)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${_svtk_build_HIERARCHY_DESTINATION}")

  get_property(_svtk_hierarchy_library_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  set(_svtk_hierarchy_filename "${_svtk_hierarchy_library_name}-hierarchy.txt")
  set(_svtk_hierarchy_file "${CMAKE_BINARY_DIR}/${_svtk_build_HIERARCHY_DESTINATION}/${_svtk_hierarchy_filename}")
  set(_svtk_hierarchy_args_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_hierarchy_library_name}-hierarchy.$<CONFIGURATION>.args")
  set(_svtk_hierarchy_data_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_hierarchy_library_name}-hierarchy.data")
  set(_svtk_hierarchy_depends_args_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_svtk_hierarchy_library_name}-hierarchy.depends.args")

  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_hierarchy" "${_svtk_hierarchy_file}")

  set(_svtk_add_module_target_name_iface "${_svtk_add_module_target_name}")
  if (_svtk_add_module_build_with_kit)
    set(_svtk_add_module_target_name_iface "${_svtk_add_module_target_name}-objects")
  endif ()
  set(_svtk_hierarchy_genex_compile_definitions
    "$<TARGET_PROPERTY:${_svtk_add_module_target_name_iface},COMPILE_DEFINITIONS>")
  set(_svtk_hierarchy_genex_include_directories
    "$<TARGET_PROPERTY:${_svtk_add_module_target_name_iface},INCLUDE_DIRECTORIES>")
  file(GENERATE
    OUTPUT  "${_svtk_hierarchy_args_file}"
    CONTENT "$<$<BOOL:${_svtk_hierarchy_genex_compile_definitions}>:\n-D\'$<JOIN:${_svtk_hierarchy_genex_compile_definitions},\'\n-D\'>\'>\n
$<$<BOOL:${_svtk_hierarchy_genex_include_directories}>:\n-I\'$<JOIN:${_svtk_hierarchy_genex_include_directories},\'\n-I\'>\'>\n")

  get_property(_svtk_hierarchy_depends_is_global GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_depends"
    SET)
  if (_svtk_hierarchy_depends_is_global)
    get_property(_svtk_hierarchy_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_depends")
  else ()
    get_property(_svtk_hierarchy_depends GLOBAL
      TARGET    "${_svtk_add_module_real_target}"
      PROPERTY  "INTERFACE_svtk_module_depends")
  endif ()

  set(_svtk_hierarchy_depends_files)
  set(_svtk_hierarchy_depends_targets)
  foreach (_svtk_hierarchy_depend IN LISTS _svtk_hierarchy_depends)
    _svtk_module_get_module_property("${_svtk_hierarchy_depend}"
      PROPERTY  "hierarchy"
      VARIABLE  _svtk_hierarchy_depend_hierarchy)
    if (NOT DEFINED _svtk_hierarchy_depend_hierarchy)
      continue ()
    endif ()

    list(APPEND _svtk_hierarchy_depends_files
      "${_svtk_hierarchy_depend_hierarchy}")

    # Find the hierarchy target of the module.
    get_property(_svtk_hierarchy_module_is_imported
      TARGET    "${_svtk_hierarchy_depend}"
      PROPERTY  IMPORTED)
    # Imported target modules are external and should already have their file
    # generated.
    if (_svtk_hierarchy_module_is_imported)
      continue ()
    endif ()

    get_property(_svtk_hierarchy_depend_library_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_hierarchy_depend}_library_name")
    if (TARGET "${_svtk_hierarchy_depend_library_name}-hierarchy")
      list(APPEND _svtk_hierarchy_depends_targets
        "${_svtk_hierarchy_depend_library_name}-hierarchy")
    endif ()
  endforeach ()

  set(_svtk_hierarchy_depends_files_arg)
  if (_svtk_hierarchy_depends_files)
    file(GENERATE
      OUTPUT  "${_svtk_hierarchy_depends_args_file}"
      CONTENT "\"$<JOIN:${_svtk_hierarchy_depends_files},\"\n\">\"\n")
  else ()
    file(GENERATE
      OUTPUT  "${_svtk_hierarchy_depends_args_file}"
      CONTENT "")
  endif ()

  _svtk_module_get_module_property("${_svtk_build_module}"
    PROPERTY  "headers"
    VARIABLE  _svtk_hierarchy_headers)
  set(_svtk_hierarchy_data_content "")
  foreach (_svtk_hierarchy_header IN LISTS _svtk_hierarchy_headers)
    string(APPEND _svtk_hierarchy_data_content
      "${_svtk_hierarchy_header};${_svtk_hierarchy_library_name}\n")
  endforeach ()
  file(GENERATE
    OUTPUT  "${_svtk_hierarchy_data_file}"
    CONTENT "${_svtk_hierarchy_data_content}")

  if (CMAKE_GENERATOR MATCHES "Ninja")
    set(_svtk_hierarchy_command_depends ${_svtk_hierarchy_depends_files})
  else ()
    set(_svtk_hierarchy_command_depends ${_svtk_hierarchy_depends_targets})
  endif ()

  set(_svtk_hierarchy_tool_target "SVTK::WrapHierarchy")
  set(_svtk_hierarchy_macros_args)
  if (TARGET SVTKCompileTools::WrapHierarchy)
    set(_svtk_hierarchy_tool_target "SVTKCompileTools::WrapHierarchy")
    if (TARGET SVTKCompileTools_macros)
      list(APPEND _svtk_hierarchy_command_depends
        "SVTKCompileTools_macros")
      list(APPEND _svtk_hierarchy_macros_args
        -undef
        -imacros "${_SVTKCompileTools_macros_file}")
    endif ()
  endif ()

  add_custom_command(
    OUTPUT  "${_svtk_hierarchy_file}"
    COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
            "$<TARGET_FILE:${_svtk_hierarchy_tool_target}>"
            "@${_svtk_hierarchy_args_file}"
            -o "${_svtk_hierarchy_file}"
            "${_svtk_hierarchy_data_file}"
            "@${_svtk_hierarchy_depends_args_file}"
            ${_svtk_hierarchy_macros_args}
    COMMENT "Generating the wrap hierarchy for ${_svtk_build_module}"
    DEPENDS
      ${_svtk_hierarchy_headers}
      "${_svtk_hierarchy_args_file}"
      "${_svtk_hierarchy_data_file}"
      "${_svtk_hierarchy_depends_args_file}"
      ${_svtk_hierarchy_command_depends})
  add_custom_target("${_svtk_add_module_library_name}-hierarchy" ALL
    DEPENDS
      "${_svtk_hierarchy_file}"
      "$<TARGET_FILE:${_svtk_hierarchy_tool_target}>")
  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_hierarchy" "${_svtk_hierarchy_file}")

  if (_svtk_build_INSTALL_HEADERS)
    set_property(TARGET "${_svtk_add_module_real_target}"
      PROPERTY
        "INTERFACE_svtk_module_hierarchy_install" "\${_svtk_module_import_prefix}/${_svtk_build_HIERARCHY_DESTINATION}/${_svtk_hierarchy_filename}")
    install(
      FILES       "${_svtk_hierarchy_file}"
      DESTINATION "${_svtk_build_HIERARCHY_DESTINATION}"
      RENAME      "${_svtk_hierarchy_filename}"
      COMPONENT   "${_svtk_build_HEADERS_COMPONENT}")
  endif ()
endfunction ()

include(GenerateExportHeader)

#[==[
@ingroup module
@brief Create a module library

~~~
svtk_module_add_module(<name>
  [FORCE_STATIC] [HEADER_ONLY]
  [EXPORT_MACRO_PREFIX      <prefix>]
  [HEADERS_SUBDIR           <subdir>]
  [LIBRARY_NAME_SUFFIX      <suffix>]
  [CLASSES                  <class>...]
  [TEMPLATE_CLASSES         <template class>...]
  [SOURCES                  <source>...]
  [HEADERS                  <header>...]
  [TEMPLATES                <template>...]
  [PRIVATE_CLASSES          <class>...]
  [PRIVATE_TEMPLATE_CLASSES <template class>...]
  [PRIVATE_HEADERS          <header>...]
  [PRIVATE_TEMPLATES        <template>...])
~~~

The `PRIVATE_` arguments are analogous to their non-`PRIVATE_` arguments, but
the associated files are not installed or available for wrapping (`SOURCES` are
always private, so there is no `PRIVATE_` variant for that argument).

  * `FORCE_STATIC`: For a static library to be created. If not provided,
    `BUILD_SHARED_LIBS` will control the library type.
  * `HEADER_ONLY`: The module only contains headers (or templates) and contains
    no compilation steps. Mutually exclusive with `FORCE_STATIC`.
  * `EXPORT_MACRO_PREFIX`: The prefix for the export macro definitions.
    Defaults to the library name of the module in all uppercase.
  * `HEADERS_SUBDIR`: The subdirectory to install headers into in the install
    tree.
  * `LIBRARY_NAME_SUFFIX`: The suffix to the module's library name if
    additional information is required.
  * `CLASSES`: A list of classes in the module. This is a shortcut for adding
    `<class>.cxx` to `SOURCES` and `<class>.h` to `HEADERS`.
  * `TEMPLATE_CLASSES`: A list of template classes in the module. This is a
    shortcut for adding `<class>.txx` to `TEMPLATES` and `<class>.h` to
    `HEADERS`.
  * `SOURCES`: A list of source files which require compilation.
  * `HEADERS`: A list of header files which will be available for wrapping and
    installed.
  * `TEMPLATES`: A list of template files which will be installed.
#]==]
function (svtk_module_add_module name)
  if (NOT name STREQUAL _svtk_build_module)
    message(FATAL_ERROR
      "The ${_svtk_build_module}'s CMakeLists.txt may not add the ${name} module.")
  endif ()

  set(_svtk_add_module_source_keywords)
  foreach (_svtk_add_module_kind IN ITEMS CLASSES TEMPLATE_CLASSES HEADERS TEMPLATES)
    list(APPEND _svtk_add_module_source_keywords
      "${_svtk_add_module_kind}"
      "PRIVATE_${_svtk_add_module_kind}")
  endforeach ()

  cmake_parse_arguments(_svtk_add_module
    "FORCE_STATIC;HEADER_ONLY"
    "EXPORT_MACRO_PREFIX;HEADERS_SUBDIR;LIBRARY_NAME_SUFFIX"
    "${_svtk_add_module_source_keywords};SOURCES"
    ${ARGN})

  if (_svtk_add_module_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_add_module: "
      "${_svtk_add_module_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_add_module_EXPORT_MACRO_PREFIX)
    _svtk_module_default_export_macro_prefix(_svtk_add_module_EXPORT_MACRO_PREFIX)
  endif ()

  if (_svtk_add_module_HEADER_ONLY AND _svtk_add_module_FORCE_STATIC)
    message(FATAL_ERROR
      "The ${_svtk_build_module} module cannot be header only yet forced "
      "static.")
  endif ()

  foreach (_svtk_add_module_class IN LISTS _svtk_add_module_CLASSES)
    list(APPEND _svtk_add_module_SOURCES
      "${_svtk_add_module_class}.cxx")
    list(APPEND _svtk_add_module_HEADERS
      "${_svtk_add_module_class}.h")
  endforeach ()

  foreach (_svtk_add_module_template_class IN LISTS _svtk_add_module_TEMPLATE_CLASSES)
    list(APPEND _svtk_add_module_TEMPLATES
      "${_svtk_add_module_template_class}.txx")
    list(APPEND _svtk_add_module_HEADERS
      "${_svtk_add_module_template_class}.h")
  endforeach ()

  foreach (_svtk_add_module_class IN LISTS _svtk_add_module_PRIVATE_CLASSES)
    list(APPEND _svtk_add_module_SOURCES
      "${_svtk_add_module_class}.cxx")
    list(APPEND _svtk_add_module_PRIVATE_HEADERS
      "${_svtk_add_module_class}.h")
  endforeach ()

  foreach (_svtk_add_module_template_class IN LISTS _svtk_add_module_PRIVATE_TEMPLATE_CLASSES)
    list(APPEND _svtk_add_module_PRIVATE_TEMPLATES
      "${_svtk_add_module_template_class}.txx")
    list(APPEND _svtk_add_module_PRIVATE_HEADERS
      "${_svtk_add_module_template_class}.h")
  endforeach ()

  if (NOT _svtk_add_module_SOURCES AND NOT _svtk_add_module_HEADER_ONLY)
    message(WARNING
      "The ${_svtk_build_module} module has no source files.")
  endif ()

  get_property(_svtk_add_module_third_party GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_third_party")

  get_property(_svtk_add_module_library_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  set(_svtk_add_module_module_header_name
    "${_svtk_add_module_library_name}Module.h")
  if (NOT _svtk_add_module_HEADER_ONLY AND NOT _svtk_add_module_third_party)
    set(_svtk_add_module_generated_header
      "${CMAKE_CURRENT_BINARY_DIR}/${_svtk_add_module_module_header_name}")
    list(APPEND _svtk_add_module_HEADERS
      "${_svtk_add_module_generated_header}")
  endif ()

  svtk_module_install_headers(
    FILES   ${_svtk_add_module_HEADERS}
            ${_svtk_add_module_TEMPLATES}
    SUBDIR  "${_svtk_add_module_HEADERS_SUBDIR}")

  set(_svtk_add_module_type)
  if (_svtk_add_module_FORCE_STATIC)
    set(_svtk_add_module_type STATIC)
  endif ()

  set(_svtk_add_module_build_with_kit)
  if (_svtk_build_BUILD_WITH_KITS)
    get_property(_svtk_add_module_build_with_kit GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_kit")
  endif ()

  get_property(_svtk_add_module_namespace GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_namespace")
  get_property(_svtk_add_module_target_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
  set(_svtk_add_module_real_target "${_svtk_add_module_target_name}")
  if (_svtk_add_module_HEADER_ONLY)
    if (_svtk_add_module_build_with_kit)
      message(FATAL_ERROR
        "The module ${_svtk_build_module} is header-only, but is part of the "
        "${_svtk_add_module_build_with_kit} kit. Header-only modules do not "
        "belong in kits.")
    endif ()

    # XXX(cmake-3.12.0): This unset is no longer necessary when 3.12.0 is required.
    unset("${_svtk_build_module}_LIB_DEPENDS" CACHE)
    add_library("${_svtk_add_module_real_target}" INTERFACE)

    if (NOT _svtk_build_module STREQUAL _svtk_add_module_real_target)
      add_library("${_svtk_build_module}" ALIAS
        "${_svtk_add_module_real_target}")
    endif ()
  else ()
    if (_svtk_add_module_build_with_kit)
      add_library("${_svtk_add_module_real_target}" INTERFACE)
      target_link_libraries("${_svtk_add_module_real_target}"
        INTERFACE
          # For usage requirements.
          "${_svtk_add_module_real_target}-objects"
          # For the implementation.
          "$<LINK_ONLY:${_svtk_add_module_build_with_kit}>")

      if (NOT _svtk_build_module STREQUAL _svtk_add_module_real_target)
        add_library("${_svtk_build_module}" ALIAS
          "${_svtk_add_module_real_target}")
      endif ()

      # Set up properties necessary for other infrastructure.
      set_property(TARGET "${_svtk_add_module_real_target}"
        PROPERTY
          "INTERFACE_svtk_module_library_name" "${_svtk_add_module_library_name}")

      # XXX(cmake-3.12.0): This unset is no longer necessary when 3.12.0 is required.
      unset("${_svtk_build_module}_LIB_DEPENDS" CACHE)
      add_library("${_svtk_add_module_real_target}-objects" OBJECT
        ${_svtk_add_module_SOURCES}
        ${_svtk_add_module_TEMPLATES}
        ${_svtk_add_module_PRIVATE_TEMPLATES}
        ${_svtk_add_module_HEADERS}
        ${_svtk_add_module_PRIVATE_HEADERS})
      set_target_properties("${_svtk_add_module_real_target}-objects"
        PROPERTIES
          # Emulate the regular library as much as possible.
          DEFINE_SYMBOL             "${_svtk_add_module_real_target}_EXPORT"
          POSITION_INDEPENDENT_CODE ON)
      target_compile_definitions("${_svtk_add_module_real_target}-objects"
        PRIVATE
          "${_svtk_add_module_real_target}_EXPORT")
      set(_svtk_add_module_real_target "${_svtk_add_module_real_target}-objects")
    else ()
      add_library("${_svtk_add_module_real_target}" ${_svtk_add_module_type}
        ${_svtk_add_module_SOURCES}
        ${_svtk_add_module_TEMPLATES}
        ${_svtk_add_module_HEADERS}
        ${_svtk_add_module_PRIVATE_HEADERS})

      set_property(TARGET "${_svtk_add_module_real_target}"
        PROPERTY
          POSITION_INDEPENDENT_CODE ON)

      if (NOT _svtk_build_module STREQUAL _svtk_add_module_real_target)
        add_library("${_svtk_build_module}" ALIAS
          "${_svtk_add_module_real_target}")
      endif ()
    endif ()
  endif ()

  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_library_name" "${_svtk_add_module_library_name}")

  get_property(_svtk_add_module_depends GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_depends")
  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_depends" "${_svtk_add_module_depends}")
  set(_svtk_add_module_includes_interface)
  if (_svtk_add_module_HEADER_ONLY)
    target_link_libraries("${_svtk_add_module_real_target}"
      INTERFACE
        ${_svtk_add_module_depends})
    set(_svtk_add_module_includes_interface INTERFACE)
  else ()
    get_property(_svtk_add_module_private_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_private_depends")

    # XXX(cmake#18484): Linking dependencies directly currently creates
    # circular dependencies. This logic should be removed once the minimum for
    # kits contains a fix for the mentioned issue.
    #
    # When two modules are part of the same kit, we can get this problem:
    #
    #   A - iface -> A-objects <- tll - K
    #   ^                               |
    #   |                               |
    #   B - iface -> B-objects <- tll -/
    #
    # If B depends on A, it ends up with a circular dependency since A has a
    # `$<LINK_ONLY:K>` link. instead, munge up dependencies of intra-kit
    # dependencies to link to the `-objects` target instead.
    if (_svtk_add_module_build_with_kit)
      set(_svtk_add_module_depends_link)
      set(_svtk_add_module_private_depends_link)
      foreach (_svtk_add_module_depend IN LISTS _svtk_add_module_depends)
        get_property(_svtk_add_module_depend_kit GLOBAL
          PROPERTY "_svtk_module_${_svtk_add_module_depend}_kit")
        if (_svtk_add_module_depend_kit STREQUAL _svtk_add_module_build_with_kit)
          # We're in the same kit; depend on the `-objects` library of the
          # module.
          get_property(_svtk_add_module_depend_target_name GLOBAL
            PROPERTY "_svtk_module_${_svtk_add_module_depend}_target_name")
          list(APPEND _svtk_add_module_depends_link
            "${_svtk_add_module_depend_target_name}-objects")
        else ()
          # Different kit, just use as normal.
          list(APPEND _svtk_add_module_depends_link
            "${_svtk_add_module_depend}")
        endif ()
      endforeach ()
      foreach (_svtk_add_module_private_depend IN LISTS _svtk_add_module_private_depends)
        get_property(_svtk_add_module_private_depend_kit GLOBAL
          PROPERTY "_svtk_module_${_svtk_add_module_private_depend}_kit")
        if (_svtk_add_module_private_depend_kit STREQUAL _svtk_add_module_build_with_kit)
          # We're in the same kit; depend on the `-objects` library of the
          # module.
          get_property(_svtk_add_module_private_depend_target_name GLOBAL
            PROPERTY "_svtk_module_${_svtk_add_module_private_depend}_target_name")
          list(APPEND _svtk_add_module_private_depends_link
            "${_svtk_add_module_private_depend_target_name}-objects")
        else ()
          # Different kit, just use as normal.
          list(APPEND _svtk_add_module_private_depends_link
            "${_svtk_add_module_private_depend}")
        endif ()
      endforeach ()

      # Add the `DEFINE_SYMBOL` for all other modules within the same kit which
      # have already been processed because the direct dependencies are not
      # sufficient: export symbols from any included header needs to be
      # correct. Since modules are built in topological order, a module can
      # only possibly include modules in the kit which have already been built.
      get_property(_svtk_add_module_kit_modules GLOBAL
        PROPERTY  "_svtk_kit_${_svtk_add_module_build_with_kit}_kit_modules")
      list(REMOVE_ITEM _svtk_add_module_kit_modules "${_svtk_build_module}")
      foreach (_svtk_add_module_kit_module IN LISTS _svtk_add_module_kit_modules)
        get_property(_svtk_add_module_kit_module_target_name GLOBAL
          PROPERTY "_svtk_module_${_svtk_add_module_kit_module}_target_name")
        if (TARGET "${_svtk_add_module_kit_module_target_name}-objects")
          get_property(_svtk_add_module_kit_module_define_symbol
            TARGET    "${_svtk_add_module_kit_module_target_name}-objects"
            PROPERTY  DEFINE_SYMBOL)
          target_compile_definitions("${_svtk_add_module_real_target}"
            PRIVATE
              "${_svtk_add_module_kit_module_define_symbol}")
        endif ()
      endforeach ()
    else ()
      set(_svtk_add_module_depends_link ${_svtk_add_module_depends})
      set(_svtk_add_module_private_depends_link ${_svtk_add_module_private_depends})
    endif ()
    target_link_libraries("${_svtk_add_module_real_target}"
      PUBLIC
        ${_svtk_add_module_depends_link}
      PRIVATE
        ${_svtk_add_module_private_depends_link})

    set(_svtk_add_module_private_depends_forward_link)
    foreach (_svtk_add_module_private_depend IN LISTS _svtk_add_module_depends_link _svtk_add_module_private_depends)
      _svtk_module_get_module_property("${_svtk_add_module_private_depend}"
        PROPERTY "forward_link"
        VARIABLE  _svtk_add_module_forward_link)
      list(APPEND _svtk_add_module_private_depends_forward_link
        ${_svtk_add_module_forward_link})
    endforeach ()

    get_property(_svtk_add_module_optional_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_optional_depends")
    foreach (_svtk_add_module_optional_depend IN LISTS _svtk_add_module_optional_depends)
      if (TARGET "${_svtk_add_module_optional_depend}")
        set(_svtk_add_module_have_optional_depend 1)
        set(_svtk_add_module_optional_depend_link "${_svtk_add_module_optional_depend}")
        if (_svtk_add_module_build_with_kit)
          get_property(_svtk_add_module_optional_depend_kit GLOBAL
            PROPERTY "_svtk_module_${_svtk_add_module_optional_depend}_kit")
          if (_svtk_add_module_optional_depend_kit STREQUAL _svtk_add_module_build_with_kit)
            # We're in the same kit; depend on the `-objects` library of the
            # module to avoid circular dependency (see explanation earlier)
            get_property(_svtk_add_module_optional_depend_target_name GLOBAL
              PROPERTY "_svtk_module_${_svtk_add_module_optional_depend}_target_name")
            set(_svtk_add_module_optional_depend_link "${_svtk_add_module_optional_depend_target_name}-objects")
          endif ()
        endif ()
        _svtk_module_get_module_property("${_svtk_add_module_optional_depend_link}"
          PROPERTY "forward_link"
          VARIABLE  _svtk_add_module_forward_link)
        list(APPEND _svtk_add_module_private_depends_forward_link
          ${_svtk_add_module_forward_link})
        target_link_libraries("${_svtk_add_module_real_target}"
          PRIVATE
            "${_svtk_add_module_optional_depend_link}")
      else ()
        set(_svtk_add_module_have_optional_depend 0)
      endif ()
      string(REPLACE "::" "_" _svtk_add_module_optional_depend_safe "${_svtk_add_module_optional_depend}")
      target_compile_definitions("${_svtk_add_module_real_target}"
        PRIVATE
          "SVTK_MODULE_ENABLE_${_svtk_add_module_optional_depend_safe}=${_svtk_add_module_have_optional_depend}")
    endforeach ()

    if (_svtk_add_module_private_depends_forward_link)
      list(REMOVE_DUPLICATES _svtk_add_module_private_depends_forward_link)
      _svtk_module_set_module_property("${_svtk_build_module}" APPEND
        PROPERTY  "forward_link"
        VALUE     "${_svtk_add_module_private_depends_forward_link}")
      target_link_libraries("${_svtk_add_module_real_target}"
        PUBLIC
          "${_svtk_add_module_private_depends_forward_link}")
    endif ()
  endif ()
  _svtk_module_standard_includes(
    TARGET  "${_svtk_add_module_real_target}"
    ${_svtk_add_module_includes_interface}
    HEADERS_DESTINATION "${_svtk_build_HEADERS_DESTINATION}")

  svtk_module_autoinit(
    MODULES ${_svtk_add_module_depends}
            ${_svtk_add_module_private_depends}
            "${_svtk_build_module}"
    TARGETS "${_svtk_add_module_real_target}")

  set(_svtk_add_module_headers_build)
  set(_svtk_add_module_headers_install)
  # TODO: Perform this in `svtk_module_install_headers` so that manually
  # installed headers may participate in wrapping as well.
  foreach (_svtk_add_module_header IN LISTS _svtk_add_module_HEADERS)
    if (IS_ABSOLUTE "${_svtk_add_module_header}")
      list(APPEND _svtk_add_module_headers_build
        "${_svtk_add_module_header}")
    else ()
      list(APPEND _svtk_add_module_headers_build
        "${CMAKE_CURRENT_SOURCE_DIR}/${_svtk_add_module_header}")
    endif ()

    get_filename_component(_svtk_add_module_header_name "${_svtk_add_module_header}" NAME)
    list(APPEND _svtk_add_module_headers_install
      "\${_svtk_module_import_prefix}/${_svtk_build_HEADERS_DESTINATION}/${_svtk_add_module_header_name}")
  endforeach ()

  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_headers" "${_svtk_add_module_headers_build}")
  if (_svtk_build_INSTALL_HEADERS)
    set_property(TARGET "${_svtk_add_module_real_target}"
      PROPERTY
        "INTERFACE_svtk_module_headers_install" "${_svtk_add_module_headers_install}")
  endif ()

  get_property(_svtk_add_module_exclude_wrap GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_exclude_wrap")
  set_property(TARGET "${_svtk_add_module_real_target}"
    PROPERTY
      "INTERFACE_svtk_module_exclude_wrap" "${_svtk_add_module_exclude_wrap}")
  if (NOT _svtk_add_module_exclude_wrap AND _svtk_build_ENABLE_WRAPPING)
    _svtk_module_write_wrap_hierarchy()
  endif ()

  set(_svtk_add_module_module_content)

  if (NOT _svtk_add_module_AUTOINIT_INCLUDE)
    get_property(_svtk_add_module_AUTOINIT_INCLUDE GLOBAL
      PROPERTY  "_svtk_module_autoinit_include")
  endif ()

  set(_svtk_add_module_autoinit_include_header)
  if (_svtk_add_module_AUTOINIT_INCLUDE)
    set(_svtk_add_module_autoinit_include_header
      "#include ${_svtk_add_module_AUTOINIT_INCLUDE}")
  endif ()

  set(_svtk_add_module_autoinit_depends_includes)
  foreach (_svtk_add_module_autoinit_dependency IN LISTS _svtk_add_module_depends)
    get_property(_svtk_add_module_autoinit_dependency_target_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_add_module_autoinit_dependency}_target_name")
    if (_svtk_add_module_autoinit_dependency_target_name)
      get_property(_svtk_add_module_depends_needs_autoinit
        TARGET    "${_svtk_add_module_autoinit_dependency_target_name}"
        PROPERTY  "INTERFACE_svtk_module_needs_autoinit")
    else ()
      set(_svtk_add_module_autoinit_dependency_target_name
        "${_svtk_add_module_autoinit_dependency}")
      get_property(_svtk_add_module_depends_needs_autoinit
        TARGET    "${_svtk_add_module_autoinit_dependency}"
        PROPERTY  "INTERFACE_svtk_module_needs_autoinit")
    endif ()
    if (NOT _svtk_add_module_depends_needs_autoinit)
      continue ()
    endif ()
    get_property(_svtk_add_module_depends_library_name
      TARGET    "${_svtk_add_module_autoinit_dependency_target_name}"
      PROPERTY  "INTERFACE_svtk_module_library_name")

    string(APPEND _svtk_add_module_autoinit_depends_includes
      "#include \"${_svtk_add_module_depends_library_name}Module.h\"\n")
  endforeach ()

  set(_svtk_add_module_autoinit_content)
  if (_svtk_add_module_autoinit_depends_includes)
    set(_svtk_add_module_autoinit_content
      "${_svtk_add_module_autoinit_content}/* AutoInit dependencies. */\n${_svtk_add_module_autoinit_depends_includes}\n")
  endif ()

  get_property(_svtk_add_module_implementable GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_implementable")
  get_property(_svtk_add_module_implements GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_implements")
  if (_svtk_add_module_implementable)
    set_property(TARGET "${_svtk_add_module_real_target}"
      PROPERTY
        "INTERFACE_svtk_module_implementable" 1)
  endif ()

  if (_svtk_add_module_implementable OR _svtk_add_module_implements)
    set_property(TARGET "${_svtk_add_module_real_target}"
      PROPERTY
        "INTERFACE_svtk_module_implements" "${_svtk_add_module_implements}")
    set_property(TARGET "${_svtk_add_module_real_target}"
      PROPERTY
        "INTERFACE_svtk_module_needs_autoinit" 1)

    set(_svtk_add_module_autoinit_content
      "${_svtk_add_module_autoinit_content}
/* AutoInit implementations. */
#ifdef ${_svtk_add_module_library_name}_AUTOINIT_INCLUDE
#include ${_svtk_add_module_library_name}_AUTOINIT_INCLUDE
#endif
#ifdef ${_svtk_add_module_library_name}_AUTOINIT
${_svtk_add_module_autoinit_include_header}
SVTK_MODULE_AUTOINIT(${_svtk_add_module_library_name})
#endif
")

    set(_svtk_add_module_module_content
      "${_svtk_add_module_module_content}${_svtk_add_module_autoinit_content}")
  endif ()

  if (NOT _svtk_add_module_HEADER_ONLY AND NOT _svtk_add_module_third_party)
    generate_export_header("${_svtk_add_module_real_target}"
      EXPORT_MACRO_NAME         "${_svtk_add_module_EXPORT_MACRO_PREFIX}_EXPORT"
      NO_EXPORT_MACRO_NAME      "${_svtk_add_module_EXPORT_MACRO_PREFIX}_NO_EXPORT"
      DEPRECATED_MACRO_NAME     "${_svtk_add_module_EXPORT_MACRO_PREFIX}_DEPRECATED"
      NO_DEPRECATED_MACRO_NAME  "${_svtk_add_module_EXPORT_MACRO_PREFIX}_NO_DEPRECATED"
      STATIC_DEFINE             "${_svtk_add_module_EXPORT_MACRO_PREFIX}_STATIC_DEFINE"
      EXPORT_FILE_NAME          "${_svtk_add_module_module_header_name}"
      CUSTOM_CONTENT_FROM_VARIABLE _svtk_add_module_module_content)
  endif ()

  _svtk_module_apply_properties("${_svtk_add_module_target_name}")
  _svtk_module_install("${_svtk_add_module_target_name}")
  _svtk_module_add_header_tests()

  if (_svtk_add_module_build_with_kit)
    _svtk_module_install("${_svtk_add_module_target_name}-objects")
  endif ()
endfunction ()

#[==[
@ingroup module-impl
@brief Add header tests for a module

@todo Move this function out to be SVTK-specific, probably into
`svtkModuleTesting.cmake`. Each module would then need to manually call this
function. It currently assumes it is in SVTK itself.

~~~
_svtk_module_add_header_tests()
~~~
#]==]
function (_svtk_module_add_header_tests)
  if (NOT BUILD_TESTING)
    return ()
  endif ()

  get_property(_svtk_add_header_tests_is_third_party GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_third_party")
  if (_svtk_add_header_tests_is_third_party)
    return ()
  endif ()

  # TODO: Add test compiles which include each header file to ensure that
  # public headers have their includes satisfied by a public dependency.

  # Bad...
  if (NOT "Python${SVTK_PYTHON_VERSION}_EXECUTABLE")
    return ()
  endif ()

  # Worse...
  if (NOT SVTK_SOURCE_DIR)
    return ()
  endif ()

  add_test(
    NAME    "${_svtk_build_module}-HeaderTest"
    COMMAND "${Python${SVTK_PYTHON_VERSION}_EXECUTABLE}"
            # TODO: What to do when using this from a SVTK install?
            "${SVTK_SOURCE_DIR}/Testing/Core/HeaderTesting.py"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${_svtk_add_module_EXPORT_MACRO}")
endfunction ()

#[==[
@ingroup module
@brief Install headers

Installing headers is done for normal modules by the @ref svtk_module_add_module
function already. However, sometimes header structures are more complicated and
need to be installed manually. This is common for third party modules or
projects which use more than a single directory of headers for a module.

To facilitate the installation of headers in various ways, the this function is
available. This function honors the `INSTALL_HEADERS`, `HEADERS_DESTINATION`,
and `HEADERS_COMPONENT` arguments to @ref svtk_module_build.

~~~
svtk_module_install_headers(
  [DIRECTORIES  <directory>...]
  [FILES        <file>...]
  [SUBDIR       <subdir>])
~~~

Installation of header directories follows CMake's `install` function semantics
with respect to trailing slashes.
#]==]
function (svtk_module_install_headers)
  cmake_parse_arguments(_svtk_install_headers
    ""
    "SUBDIR"
    "FILES;DIRECTORIES"
    ${ARGN})

  if (_svtk_install_headers_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_install_headers: "
      "${_svtk_install_headers_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT _svtk_build_INSTALL_HEADERS)
    return ()
  endif ()

  if (NOT _svtk_install_headers_FILES AND NOT _svtk_install_headers_DIRECTORIES)
    return ()
  endif ()

  set(_svtk_install_headers_destination
    "${_svtk_build_HEADERS_DESTINATION}/${_svtk_install_headers_SUBDIR}")
  if (_svtk_install_headers_FILES)
    install(
      FILES       ${_svtk_install_headers_FILES}
      DESTINATION "${_svtk_install_headers_destination}"
      COMPONENT   "${_svtk_build_HEADERS_COMPONENT}")
  endif ()
  foreach (_svtk_install_headers_directory IN LISTS _svtk_install_headers_DIRECTORIES)
    install(
      DIRECTORY   "${_svtk_install_headers_directory}"
      DESTINATION "${_svtk_install_headers_destination}"
      COMPONENT   "${_svtk_build_HEADERS_COMPONENT}")
  endforeach ()
endfunction ()

#[==[
@ingroup module-internal
@brief Apply properties to a module

Apply build properties to a target. Generally only useful to wrapping code or
other modules that cannot use @ref svtk_module_add_module for some reason.

~~~
_svtk_module_apply_properties(<target>
  [BASENAME <basename>])
~~~

If `BASENAME` is given, it will be used instead of the target name as the basis
for `OUTPUT_NAME`. Full modules (as opposed to third party or other non-module
libraries) always use the module's `LIBRARY_NAME` setting.

The following target properties are set based on the arguments to the calling
@ref svtk_module_build call:

  - `OUTPUT_NAME` (based on the module's `LIBRARY_NAME` and
    `svtk_module_build(LIBRARY_NAME_SUFFIX)`)
  - `VERSION` (based on `svtk_module_build(VERSION)`)
  - `SOVERSION` (based on `svtk_module_build(SOVERSION)`)
  - `DEBUG_POSTFIX` (on Windows)
#]==]
function (_svtk_module_apply_properties target)
  cmake_parse_arguments(_svtk_apply_properties
    ""
    "BASENAME"
    ""
    ${ARGN})

  if (_svtk_apply_properties_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for _svtk_module_apply_properties: "
      "${_svtk_apply_properties_UNPARSED_ARGUMENTS}.")
  endif ()

  if (NOT DEFINED _svtk_apply_properties_BASENAME)
    set(_svtk_apply_properties_BASENAME "${target}")
  endif ()

  get_property(_svtk_add_module_type
    TARGET    "${target}"
    PROPERTY  TYPE)
  if (_svtk_add_module_type STREQUAL "OBJECT_LIBRARY" OR
      _svtk_add_module_type STREQUAL "INTERFACE_LIBRARY")
    return ()
  endif ()

  set(_svtk_add_module_library_name "${_svtk_apply_properties_BASENAME}")
  get_property(_svtk_add_module_target_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
  if (_svtk_add_module_target_name STREQUAL "${target}")
    get_property(_svtk_add_module_library_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  endif ()
  set(_svtk_add_module_output_name "${_svtk_add_module_library_name}${_svtk_add_module_LIBRARY_NAME_SUFFIX}")
  if (_svtk_build_LIBRARY_NAME_SUFFIX)
    string(APPEND _svtk_add_module_output_name "-${_svtk_build_LIBRARY_NAME_SUFFIX}")
  endif ()

  set_target_properties("${target}"
    PROPERTIES
      OUTPUT_NAME "${_svtk_add_module_output_name}")

  if (_svtk_build_VERSION AND NOT _svtk_add_module_type STREQUAL "EXECUTABLE")
    set_target_properties("${target}"
      PROPERTIES
        VERSION "${_svtk_build_VERSION}")
  endif ()

  if (_svtk_build_SOVERSION)
    set_target_properties("${target}"
      PROPERTIES
        SOVERSION "${_svtk_build_SOVERSION}")
  endif ()

  if (WIN32)
    set_target_properties("${target}"
      PROPERTIES
        DEBUG_POSTFIX "d")
  endif ()
endfunction ()

#[==[
@ingroup module-internal
@brief Install a module target

Install a target within the module context. Generally only useful to wrapping
code, modules that cannot use @ref svtk_module_add_module for some reason, or
modules which create utility targets that need installed.

~~~
_svtk_module_install(<target>)
~~~

This function uses the various installation options to @ref svtk_module_build
function to keep the install uniform.
#]==]
function (_svtk_module_install target)
  set(_svtk_install_export)
  if (_svtk_build_INSTALL_EXPORT)
    set(_svtk_install_export
      EXPORT "${_svtk_build_INSTALL_EXPORT}")
  endif ()

  set(_svtk_install_namelink_args)
  if(NOT CMAKE_VERSION VERSION_LESS 3.12)
    list(APPEND _svtk_install_namelink_args
      NAMELINK_COMPONENT "${_svtk_build_HEADERS_COMPONENT}")
  endif()
  install(
    TARGETS             "${target}"
    ${_svtk_install_export}
    ${ARGN}
    ARCHIVE
      DESTINATION "${_svtk_build_ARCHIVE_DESTINATION}"
      COMPONENT   "${_svtk_build_HEADERS_COMPONENT}"
    LIBRARY
      DESTINATION "${_svtk_build_LIBRARY_DESTINATION}"
      COMPONENT   "${_svtk_build_TARGETS_COMPONENT}"
      ${_svtk_install_namelink_args}
    RUNTIME
      DESTINATION "${_svtk_build_RUNTIME_DESTINATION}"
      COMPONENT   "${_svtk_build_TARGETS_COMPONENT}")
endfunction ()

#[==[
@ingroup module
@brief Create a module executable

Some modules may have associated executables with them. By using this function,
the target will be installed following the options given to the associated
@ref svtk_module_build command. Its name will also be changed according to the
`LIBRARY_NAME_SUFFIX` option.

~~~
svtk_module_add_executable(<name>
  [NO_INSTALL]
  [DEVELOPMENT]
  [BASENAME <basename>]
  <source>...)
~~~

If `NO_INSTALL` is specified, the executable will not be installed. If
`BASENAME` is given, it will be used as the name of the executable rather than
the target name.

If `DEVELOPMENT` is given, it marks the executable as a development tool and
will not be installed if `INSTALL_HEADERS` is not set for the associated
@ref svtk_module_build command.

If the executable being built is the module, its module properties are used
rather than `BASENAME`. In addition, the dependencies of the module will be
linked.
#]==]
function (svtk_module_add_executable name)
  cmake_parse_arguments(_svtk_add_executable
    "NO_INSTALL;DEVELOPMENT"
    "BASENAME"
    ""
    ${ARGN})

  if (NOT _svtk_add_executable_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "The ${name} executable must have at least one source file.")
  endif ()

  if (_svtk_add_executable_NO_INSTALL AND _svtk_add_executable_DEVELOPMENT)
    message(FATAL_ERROR
      "Both `NO_INSTALL` and `DEVELOPMENT` may not be specified.")
  endif ()

  set(_svtk_add_executable_target_name "${name}")
  set(_svtk_add_executable_library_name "${name}")
  if (name STREQUAL _svtk_build_module)
    if (_svtk_add_executable_NO_INSTALL)
      message(FATAL_ERROR
        "The executable ${_svtk_build_module} module may not use `NO_INSTALL`.")
    endif ()
    if (DEFINED _svtk_add_executable_BASENAME)
      message(FATAL_ERROR
        "The executable ${_svtk_build_module} module may not pass `BASENAME` "
        "when adding the executable; it is controlled via `LIBRARY_NAME` in "
        "the associated `svtk.module` file.")
    endif ()
    get_property(_svtk_add_executable_target_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
    get_property(_svtk_add_executable_library_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  endif ()

  if (_svtk_add_executable_DEVELOPMENT AND NOT _svtk_build_INSTALL_HEADERS)
    set(_svtk_add_executable_NO_INSTALL ON)
  endif ()

  # Set up rpaths
  set(CMAKE_BUILD_RPATH_USE_ORIGIN 1)
  if (UNIX)
    file(RELATIVE_PATH _svtk_add_executable_relpath
      "/prefix/${_svtk_build_RUNTIME_DESTINATION}"
      "/prefix/${_svtk_build_LIBRARY_DESTINATION}")
    if (APPLE)
      set(_svtk_add_executable_origin_rpath_prefix
        "@executable_path")
    else ()
      set(_svtk_add_executable_origin_rpath_prefix
        "$ORIGIN")
    endif ()

    list(APPEND CMAKE_INSTALL_RPATH
      "${_svtk_add_executable_origin_rpath_prefix}/${_svtk_add_executable_relpath}")
  endif ()

  add_executable("${_svtk_add_executable_target_name}"
    ${_svtk_add_executable_UNPARSED_ARGUMENTS})

  if (name STREQUAL _svtk_build_module AND NOT _svtk_add_executable_target_name STREQUAL _svtk_build_module)
    add_executable("${_svtk_build_module}" ALIAS
      "${_svtk_add_executable_target_name}")
  endif ()

  if (name STREQUAL _svtk_build_module)
    get_property(_svtk_real_target_kit GLOBAL
      PROPERTY "_svtk_module_${_svtk_build_module}_kit")
    if (_svtk_real_target_kit)
      message(FATAL_ERROR
        "Executable module ${_svtk_build_module} is declared to be part of a "
        "kit; this is not possible.")
    endif ()

    get_property(_svtk_add_executable_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_depends")
    get_property(_svtk_add_executable_private_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_private_depends")
    target_link_libraries("${_svtk_add_executable_target_name}"
      PUBLIC
        ${_svtk_add_executable_depends}
      PRIVATE
        ${_svtk_add_executable_private_depends})
    get_property(_svtk_add_executable_optional_depends GLOBAL
      PROPERTY  "_svtk_module_${_svtk_build_module}_optional_depends")
    foreach (_svtk_add_executable_optional_depend IN LISTS _svtk_add_executable_optional_depends)
      string(REPLACE "::" "_" _svtk_add_executable_optional_depend_safe "${_svtk_add_executable_optional_depend}")
      if (TARGET "${_svtk_add_executable_optional_depend}")
        set(_svtk_add_executable_have_optional_depend 1)
      else ()
        set(_svtk_add_executable_have_optional_depend 0)
      endif ()
      target_compile_definitions("${_svtk_add_executable_target_name}"
        PRIVATE
          "SVTK_MODULE_ENABLE_${_svtk_add_executable_optional_depend_safe}=${_svtk_add_executable_have_optional_depend}")
    endforeach ()

    if (_svtk_module_warnings)
      if (_svtk_add_executable_depends)
        message(WARNING
          "Executable module ${_svtk_build_module} has public dependencies; this "
          "shouldn't be necessary.")
      endif ()
    endif ()
  endif ()

  set(_svtk_add_executable_property_args)
  if (DEFINED _svtk_add_executable_BASENAME)
    list(APPEND _svtk_add_executable_property_args
      BASENAME "${_svtk_add_executable_BASENAME}")
  endif ()

  _svtk_module_apply_properties("${_svtk_add_executable_target_name}"
    ${_svtk_add_executable_property_args})
  _svtk_module_standard_includes(TARGET "${_svtk_add_executable_target_name}")

  if (NOT _svtk_add_executable_NO_INSTALL)
    _svtk_module_install("${_svtk_add_executable_target_name}")
  endif ()
endfunction ()

#[==[
@ingroup module
@brief Find a package

A wrapper around `find_package` that records information for use so that the
same targets may be found when finding this package.

Modules may need to find external dependencies. CMake often provides modules to
find these dependencies, but when imported targets are involved, these.need to
also be found from dependencies of the current project. Since the benefits of
imported targets greatly outweighs not using them, it is preferred to use them.

The module system provides the @ref svtk_module_find_package function in order
to extend `find_package` support to include finding the dependencies from an
install of the project.

~~~
svtk_module_find_package(
  [PRIVATE] [CONFIG_MODE]
  PACKAGE               <package>
  [VERSION              <version>]
  [COMPONENTS           <component>...]
  [OPTIONAL_COMPONENTS  <component>...]
  [FORWARD_VERSION_REQ  <MAJOR|MINOR|PATCH|EXACT>]
  [VERSION_VAR          <variable>])
~~~

  * `PACKAGE`: The name of the package to find.
  * `VERSION`: The minimum version of the package that is required.
  * `COMPONENTS`: Components of the package which are required.
  * `OPTIONAL_COMPONENTS`: Components of the package which may be missing.
  * `FORWARD_VERSION_REQ`: If provided, the found version will be promoted to
    the minimum version required matching the given version scheme.
  * `VERSION_VAR`: The variable to use as the provided version (defaults to
    `<PACKAGE>_VERSION`). It may contain `@` in which case it will be
    configured. This is useful for modules which only provide components of the
    actual version number.
  * `CONFIG_MODE`: If present, pass `CONFIG` to the underlying `find_package`
    call.
  * `PRIVATE`: The dependency should not be exported to the install.

The `PACKAGE` argument is the only required argument. The rest are optional.

Note that `PRIVATE` is *only* applicable for private dependencies on interface
targets (basically, header libraries) because some platforms require private
shared libraries dependencies to be present when linking dependent libraries
and executables as well.
#]==]
macro (svtk_module_find_package)
  # This needs to be a macro because find modules typically set variables which
  # may need to be available in the calling scope. If we declare that it only
  # works with imported targets (which is the primary motivating factor behind
  # this function), we can instead make it a function at the cost of any
  # non-target variables a module might want to set being available. It is
  # unlikely that this will be the case for all callers.
  if (NOT _svtk_build_module)
    message(FATAL_ERROR
      "`svtk_module_find_package` may only be called when building a SVTK "
      "module.")
  endif ()

  # Note: when adding arguments here, add them to the `unset` block at the end
  # of the function.
  cmake_parse_arguments(_svtk_find_package
    "PRIVATE;CONFIG_MODE"
    "PACKAGE;VERSION;FORWARD_VERSION_REQ;VERSION_VAR"
    "COMPONENTS;OPTIONAL_COMPONENTS"
    ${ARGN})

  if (_svtk_find_package_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_find_package: "
      "${_svtk_find_package_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_find_package_PACKAGE)
    message(FATAL_ERROR
      "The `PACKAGE` argument is required.")
  endif ()

  if (DEFINED _svtk_find_package_FORWARD_VERSION_REQ)
    if (_svtk_find_package_PRIVATE)
      message(FATAL_ERROR
        "The `FORWARD_VERSION_REQ` argument is incompatible with the "
        "`PRIVATE` flag.")
    endif ()

    if (NOT _svtk_find_package_FORWARD_VERSION_REQ STREQUAL "MAJOR" AND
        NOT _svtk_find_package_FORWARD_VERSION_REQ STREQUAL "MINOR" AND
        NOT _svtk_find_package_FORWARD_VERSION_REQ STREQUAL "PATCH" AND
        NOT _svtk_find_package_FORWARD_VERSION_REQ STREQUAL "EXACT")
      message(FATAL_ERROR
        "The `FORWARD_VERSION_REQ` argument must be one of `MAJOR`, `MINOR`, "
        "`PATCH`, or `EXACT`.")
    endif ()
  endif ()

  if (NOT DEFINED _svtk_find_package_VERSION_VAR)
    set(_svtk_find_package_VERSION_VAR
      "${_svtk_find_package_PACKAGE}_VERSION")
  endif ()

  set(_svtk_find_package_config)
  if (_svtk_find_package_CONFIG_MODE)
    set(_svtk_find_package_config "CONFIG")
  endif ()

  find_package("${_svtk_find_package_PACKAGE}"
    ${_svtk_find_package_VERSION}
    ${_svtk_find_package_config}
    COMPONENTS          ${_svtk_find_package_COMPONENTS}
    OPTIONAL_COMPONENTS ${_svtk_find_package_OPTIONAL_COMPONENTS})
  if (NOT ${_svtk_find_package_PACKAGE}_FOUND)
    message(FATAL_ERROR
      "Could not find the ${_svtk_find_package_PACKAGE} external dependency.")
    return ()
  endif ()

  set(_svtk_find_package_optional_components_found)
  foreach (_svtk_find_package_optional_component IN LISTS _svtk_find_package_OPTIONAL_COMPONENTS)
    if (${_svtk_find_package_PACKAGE}_${_svtk_find_package_optional_component}_FOUND)
      list(APPEND _svtk_find_package_optional_components_found
        "${_svtk_find_package_optional_component}")
    endif ()
  endforeach ()

  if (NOT _svtk_find_package_PRIVATE)
    set_property(GLOBAL APPEND
      PROPERTY
        "_svtk_module_find_packages_${_svtk_build_PACKAGE}" "${_svtk_find_package_PACKAGE}")
    set(_svtk_find_package_base "_svtk_module_find_package_${_svtk_build_module}")
    set_property(GLOBAL APPEND
      PROPERTY
        "${_svtk_find_package_base}" "${_svtk_find_package_PACKAGE}")
    set(_svtk_find_package_base_package "${_svtk_find_package_base}_${_svtk_find_package_PACKAGE}")
    set_property(GLOBAL
      PROPERTY
        "${_svtk_find_package_base_package}_version" "${_svtk_find_package_VERSION}")
    set_property(GLOBAL
      PROPERTY
        "${_svtk_find_package_base_package}_config" "${_svtk_find_package_CONFIG_MODE}")
    set_property(GLOBAL APPEND
      PROPERTY
        "${_svtk_find_package_base_package}_components" "${_svtk_find_package_COMPONENTS}")
    set_property(GLOBAL APPEND
      PROPERTY
        "${_svtk_find_package_base_package}_optional_components" "${_svtk_find_package_OPTIONAL_COMPONENTS}")
    set_property(GLOBAL APPEND
      PROPERTY
        "${_svtk_find_package_base_package}_optional_components_found" "${_svtk_find_package_optional_components_found}")
    set_property(GLOBAL
      PROPERTY
        "${_svtk_find_package_base_package}_exact" "0")
    if (DEFINED _svtk_find_package_FORWARD_VERSION_REQ)
      string(FIND "${_svtk_find_package_VERSION_VAR}" "@" _svtk_find_package_idx)
      if (_svtk_find_package_idx EQUAL -1)
        if (NOT DEFINED "${_svtk_find_package_VERSION_VAR}")
          message(FATAL_ERROR
            "The `${_svtk_find_package_VERSION_VAR}` variable is not defined.")
        endif ()
        set(_svtk_find_package_version "${${_svtk_find_package_VERSION_VAR}}")
      else ()
        string(CONFIGURE "${_svtk_find_package_VERSION_VAR}" _svtk_find_package_version)
      endif ()
      unset(_svtk_find_package_idx)

      if ("${_svtk_find_package_version}" STREQUAL "")
        message(FATAL_ERROR
          "The `${_svtk_find_package_PACKAGE}` version is empty.")
      endif ()

      if (_svtk_find_package_FORWARD_VERSION_REQ STREQUAL "MAJOR")
        set(_svtk_find_package_version_regex "^\([^.]*\).*")
      elseif (_svtk_find_package_FORWARD_VERSION_REQ STREQUAL "MINOR")
        set(_svtk_find_package_version_regex "^\([^.]*.[^.]*\).*")
      elseif (_svtk_find_package_FORWARD_VERSION_REQ STREQUAL "PATCH")
        set(_svtk_find_package_version_regex "^\([^.]*.[^.]*.[^.]*\).*")
      elseif (_svtk_find_package_FORWARD_VERSION_REQ STREQUAL "EXACT")
        set(_svtk_find_package_version_regex "^\\(.*\\)$")
        set_property(GLOBAL
          PROPERTY
            "${_svtk_find_package_base_package}_exact" "1")
      endif ()

      string(REGEX REPLACE "${_svtk_find_package_version_regex}" "\\1"
        _svtk_find_package_found_version "${_svtk_find_package_version}")
      unset(_svtk_find_package_version_regex)
      unset(_svtk_find_package_version)

      set_property(GLOBAL
        PROPERTY
          "${_svtk_find_package_base_package}_version" "${_svtk_find_package_found_version}")
      unset(_svtk_find_package_found_version)
    endif ()
  endif ()

  unset(_svtk_find_package_base)
  unset(_svtk_find_package_base_package)
  unset(_svtk_find_package_COMPONENTS)
  unset(_svtk_find_package_FORWARD_VERSION_REQ)
  unset(_svtk_find_package_OPTIONAL_COMPONENTS)
  unset(_svtk_find_package_PACKAGE)
  unset(_svtk_find_package_PRIVATE)
  unset(_svtk_find_package_UNPARSED_ARGUMENTS)
  unset(_svtk_find_package_VERSION)
  unset(_svtk_find_package_VERSION_VAR)
endmacro ()

#[==[
@ingroup module
@brief Export find_package calls for dependencies

When installing a project that is meant to be found via `find_package` from
CMake, using imported targets in the build means that imported targets need to
be created during the `find_package` as well. This function writes a file
suitable for inclusion from a `<package>-config.cmake` file to satisfy
dependencies. It assumes that the exported targets are named
`${CMAKE_FIND_PACKAGE_NAME}::${component}`. Dependent packages will only be
found if a requested component requires the package to be found either directly
or transitively.

~~~
svtk_module_export_find_packages(
  CMAKE_DESTINATION <directory>
  FILE_NAME         <filename>
  [COMPONENT        <component>]
  MODULES           <module>...)
~~~

The file will be named according to the `FILE_NAME` argument will be installed
into `CMAKE_DESTINATION` in the build and install trees with the given
filename. If not provided, the `development` component will be used.

The `svtk_module_find_package` calls made by the modules listed in `MODULES`
will be exported to this file.
#]==]
function (svtk_module_export_find_packages)
  cmake_parse_arguments(_svtk_export
    ""
    "CMAKE_DESTINATION;FILE_NAME;COMPONENT"
    "MODULES"
    ${ARGN})

  if (_svtk_export_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_export_find_packages: "
      "${_svtk_export_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_export_CMAKE_DESTINATION)
    message(FATAL_ERROR
      "The `CMAKE_DESTINATION` is required.")
  endif ()

  if (NOT DEFINED _svtk_export_FILE_NAME)
    message(FATAL_ERROR
      "The `FILE_NAME` is required.")
  endif ()

  if (NOT DEFINED _svtk_export_COMPONENT)
    set(_svtk_export_COMPONENT "development")
  endif ()

  set(_svtk_export_output_file
    "${CMAKE_BINARY_DIR}/${_svtk_export_CMAKE_DESTINATION}/${_svtk_export_FILE_NAME}")
  file(WRITE "${_svtk_export_output_file}"
"set(_svtk_module_find_package_quiet)
if (\${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_svtk_module_find_package_quiet QUIET)
endif ()

set(_svtk_module_find_package_components_checked)
set(_svtk_module_find_package_components_to_check
  \${\${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})
set(_svtk_module_find_package_components)
set(_svtk_module_find_package_components_required)
while (_svtk_module_find_package_components_to_check)
  list(GET _svtk_module_find_package_components_to_check 0 _svtk_module_component)
  list(REMOVE_AT _svtk_module_find_package_components_to_check 0)
  if (_svtk_module_component IN_LIST _svtk_module_find_package_components_checked)
    continue ()
  endif ()
  list(APPEND _svtk_module_find_package_components_checked
    \"\${_svtk_module_component}\")

  list(APPEND _svtk_module_find_package_components
    \"\${_svtk_module_component}\")
  if (\${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_\${_svtk_module_component})
    list(APPEND _svtk_module_find_package_components_required
      \"\${_svtk_module_component}\")
  endif ()

  if (TARGET \"\${CMAKE_FIND_PACKAGE_NAME}::\${_svtk_module_component}\")
    set(_svtk_module_find_package_component_target \"\${CMAKE_FIND_PACKAGE_NAME}::\${_svtk_module_component}\")
  elseif (TARGET \"\${_svtk_module_component}\")
    set(_svtk_module_find_package_component_target \"\${_svtk_module_component}\")
  else ()
    # No such target for the component; skip.
    continue ()
  endif ()
  get_property(_svtk_module_find_package_depends
    TARGET    \"\${_svtk_module_find_package_component_target}\"
    PROPERTY  \"INTERFACE_svtk_module_depends\")
  string(REPLACE \"\${CMAKE_FIND_PACKAGE_NAME}::\" \"\" _svtk_module_find_package_depends \"\${_svtk_module_find_package_depends}\")
  list(APPEND _svtk_module_find_package_components_to_check
    \${_svtk_module_find_package_depends})
  get_property(_svtk_module_find_package_depends
    TARGET    \"\${_svtk_module_find_package_component_target}\"
    PROPERTY  \"INTERFACE_svtk_module_private_depends\")
  string(REPLACE \"\${CMAKE_FIND_PACKAGE_NAME}::\" \"\" _svtk_module_find_package_depends \"\${_svtk_module_find_package_depends}\")
  list(APPEND _svtk_module_find_package_components_to_check
    \${_svtk_module_find_package_depends})
  get_property(_svtk_module_find_package_depends
    TARGET    \"\${_svtk_module_find_package_component_target}\"
    PROPERTY  \"INTERFACE_svtk_module_optional_depends\")
  foreach (_svtk_module_find_package_depend IN LISTS _svtk_module_find_package_depends)
    if (TARGET \"\${_svtk_module_find_package_depend}\")
      string(REPLACE \"\${CMAKE_FIND_PACKAGE_NAME}::\" \"\" _svtk_module_find_package_depend \"\${_svtk_module_find_package_depend}\")
      list(APPEND _svtk_module_find_package_components_to_check
        \"\${_svtk_module_find_package_depend}\")
    endif ()
  endforeach ()
  get_property(_svtk_module_find_package_depends
    TARGET    \"\${_svtk_module_find_package_component_target}\"
    PROPERTY  \"INTERFACE_svtk_module_forward_link\")
  string(REPLACE \"\${CMAKE_FIND_PACKAGE_NAME}::\" \"\" _svtk_module_find_package_depends \"\${_svtk_module_find_package_depends}\")
  list(APPEND _svtk_module_find_package_components_to_check
    \${_svtk_module_find_package_depends})

  get_property(_svtk_module_find_package_kit
    TARGET    \"\${_svtk_module_find_package_component_target}\"
    PROPERTY  \"INTERFACE_svtk_module_kit\")
  if (_svtk_module_find_package_kit)
    get_property(_svtk_module_find_package_kit_modules
      TARGET    \"\${_svtk_module_find_package_kit}\"
      PROPERTY  \"INTERFACE_svtk_kit_kit_modules\")
    string(REPLACE \"\${CMAKE_FIND_PACKAGE_NAME}::\" \"\" _svtk_module_find_package_kit_modules \"\${_svtk_module_find_package_kit_modules}\")
    list(APPEND _svtk_module_find_package_components_to_check
      \${_svtk_module_find_package_kit_modules})
  endif ()
endwhile ()
unset(_svtk_module_find_package_component_target)
unset(_svtk_module_find_package_components_to_check)
unset(_svtk_module_find_package_components_checked)
unset(_svtk_module_component)
unset(_svtk_module_find_package_depend)
unset(_svtk_module_find_package_depends)
unset(_svtk_module_find_package_kit)
unset(_svtk_module_find_package_kit_modules)

if (_svtk_module_find_package_components)
  list(REMOVE_DUPLICATES _svtk_module_find_package_components)
endif ()
if (_svtk_module_find_package_components_required)
  list(REMOVE_DUPLICATES _svtk_module_find_package_components_required)
endif ()\n\n")

  foreach (_svtk_export_module IN LISTS _svtk_export_MODULES)
    get_property(_svtk_export_target_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_export_module}_target_name")
    set(_svtk_export_base "_svtk_module_find_package_${_svtk_export_module}")
    get_property(_svtk_export_packages GLOBAL
      PROPERTY "${_svtk_export_base}")
    if (NOT _svtk_export_packages)
      continue ()
    endif ()

    file(APPEND "${_svtk_export_output_file}"
"set(_svtk_module_find_package_enabled OFF)
set(_svtk_module_find_package_is_required OFF)
set(_svtk_module_find_package_fail_if_not_found OFF)
if (_svtk_module_find_package_components)
  if (\"${_svtk_export_target_name}\" IN_LIST _svtk_module_find_package_components)
    set(_svtk_module_find_package_enabled ON)
    if (\"${_svtk_export_target_name}\" IN_LIST _svtk_module_find_package_components_required)
      set(_svtk_module_find_package_is_required \"\${\${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED}\")
      set(_svtk_module_find_package_fail_if_not_found ON)
    endif ()
  endif ()
else ()
  set(_svtk_module_find_package_enabled ON)
  set(_svtk_module_find_package_is_required \"\${\${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED}\")
  set(_svtk_module_find_package_fail_if_not_found ON)
endif ()

if (_svtk_module_find_package_enabled)
  set(_svtk_module_find_package_required)
  if (_svtk_module_find_package_is_required)
    set(_svtk_module_find_package_required REQUIRED)
  endif ()\n\n")

    list(REMOVE_DUPLICATES _svtk_export_packages)
    foreach (_svtk_export_package IN LISTS _svtk_export_packages)
      set(_svtk_export_base_package "${_svtk_export_base}_${_svtk_export_package}")
      get_property(_svtk_export_version GLOBAL
        PROPERTY "${_svtk_export_base_package}_version")
      get_property(_svtk_export_config GLOBAL
        PROPERTY "${_svtk_export_base_package}_config")
      get_property(_svtk_export_exact GLOBAL
        PROPERTY "${_svtk_export_base_package}_exact")
      get_property(_svtk_export_components GLOBAL
        PROPERTY "${_svtk_export_base_package}_components")
      get_property(_svtk_export_optional_components GLOBAL
        PROPERTY "${_svtk_export_base_package}_optional_components")
      get_property(_svtk_export_optional_components_found GLOBAL
        PROPERTY "${_svtk_export_base_package}_optional_components_found")

      # Assume that any found optional components end up being required.
      if (${_svtk_export_base_package}_optional_components_found)
        list(REMOVE_ITEM _svtk_export_optional_components
          ${_svtk_export_optional_components_found})
        list(APPEND _svtk_export_components
          ${_svtk_export_optional_components_found})
      endif ()

      set(_svtk_export_config_arg)
      if (_svtk_export_config)
        set(_svtk_export_config_arg CONFIG)
      endif ()

      set(_svtk_export_exact_arg)
      if (_svtk_export_exact)
        set(_svtk_export_exact_arg EXACT)
      endif ()

      file(APPEND "${_svtk_export_output_file}"
"  find_package(${_svtk_export_package}
    ${_svtk_export_version}
    ${_svtk_export_exact_arg}
    ${_svtk_export_config_arg}
    \${_svtk_module_find_package_quiet}
    \${_svtk_module_find_package_required}
    COMPONENTS          ${_svtk_export_components}
    OPTIONAL_COMPONENTS ${_svtk_export_optional_components})
  if (NOT ${_svtk_export_package}_FOUND AND _svtk_module_find_package_fail_if_not_found)
    if (NOT \${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
      message(STATUS
        \"Could not find the \${CMAKE_FIND_PACKAGE_NAME} package due to a \"
        \"missing dependency: ${_svtk_export_package}\")
    endif ()
    set(\"\${CMAKE_FIND_PACKAGE_NAME}_${_svtk_export_target_name}_FOUND\" 0)
    list(APPEND \"\${CMAKE_FIND_PACKAGE_NAME}_${_svtk_export_target_name}_NOT_FOUND_MESSAGE\"
      \"Failed to find the ${_svtk_export_package} package.\")
  endif ()\n")
    endforeach ()

    file(APPEND "${_svtk_export_output_file}"
"endif ()

unset(_svtk_module_find_package_fail_if_not_found)
unset(_svtk_module_find_package_enabled)
unset(_svtk_module_find_package_required)\n\n")

  endforeach ()

  file(APPEND "${_svtk_export_output_file}"
    "unset(_svtk_module_find_package_components)
unset(_svtk_module_find_package_components_required)
unset(_svtk_module_find_package_quiet)\n")

  install(
    FILES       "${CMAKE_BINARY_DIR}/${_svtk_export_CMAKE_DESTINATION}/${_svtk_export_FILE_NAME}"
    DESTINATION "${_svtk_export_CMAKE_DESTINATION}"
    COMPONENT   "${_svtk_export_COMPONENT}")
endfunction ()

#[==[
@page module-overview

@ingroup module
@section module-third-party Third party support

The module system acknowledges that third party support is a pain and offers
APIs to help wrangle them. Sometimes third party code needs a shim introduced
to make it behave better, so an `INTERFACE` library to add that in is very
useful. Other times, third party code is hard to ensure that it exists
everywhere, so it is bundled. When that happens, the ability to select between
the bundled copy and an external copy is useful. All three (and more) of these
are possible.

The following functions are used to handle third party modules:

  - @ref svtk_module_third_party
  - @ref svtk_module_third_party_external
  - @ref svtk_module_third_party_internal
#]==]

#[==[
@ingroup module
@brief Third party module

When a project has modules which represent third party packages, there are some
convenience functions to help deal with them. First, there is the meta-wrapper:

~~~
svtk_module_third_party(
  [INTERNAL <internal arguments>...]
  [EXTERNAL <external arguments>...])
~~~

This offers a cache variable named `SVTK_MODULE_USE_EXTERNAL_<module name>` that
may be set to trigger between the internal copy and an externally provided
copy. This is available as a local variable named
`SVTK_MODULE_USE_EXTERNAL_<library name>`. See the
@ref svtk_module_third_party_external and @ref svtk_module_third_party_internal
functions for the arguments supported by the `EXTERNAL` and `INTERNAL`
arguments, respectively.
#]==]
function (svtk_module_third_party)
  cmake_parse_arguments(_svtk_third_party
    ""
    ""
    "INTERNAL;EXTERNAL"
    ${ARGN})

  if (_svtk_third_party_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_third_party: "
      "${_svtk_third_party_UNPARSED_ARGUMENTS}")
  endif ()

  string(REPLACE "::" "_" _svtk_build_module_safe "${_svtk_build_module}")
  option("SVTK_MODULE_USE_EXTERNAL_${_svtk_build_module_safe}"
    "Use externally provided ${_svtk_build_module}"
    "${_svtk_build_USE_EXTERNAL}")
  mark_as_advanced("SVTK_MODULE_USE_EXTERNAL_${_svtk_build_module_safe}")
  get_property(_svtk_third_party_library_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  set("SVTK_MODULE_USE_EXTERNAL_${_svtk_third_party_library_name}"
    "${SVTK_MODULE_USE_EXTERNAL_${_svtk_build_module_safe}}"
    PARENT_SCOPE)

  if (SVTK_MODULE_USE_EXTERNAL_${_svtk_build_module_safe})
    # XXX(cmake): https://gitlab.kitware.com/cmake/cmake/issues/16364.
    # Unset a variable which CMake doesn't like when switching between real
    # libraries (internal) and interface libraries (external).
    unset("${_svtk_build_module}_LIB_DEPENDS" CACHE)
    svtk_module_third_party_external(${_svtk_third_party_EXTERNAL})

    # Bubble up variables again.
    foreach (_svtk_third_party_variable IN LISTS _svtk_third_party_variables)
      set("${_svtk_third_party_variable}"
        "${${_svtk_third_party_variable}}"
        PARENT_SCOPE)
    endforeach ()
  else ()
    set(_svtk_third_party_has_external_support 1)
    svtk_module_third_party_internal(${_svtk_third_party_INTERNAL})
  endif ()
endfunction ()

#[==[
@ingroup module-impl
@brief Mark a module as being third party

Mark a module as being a third party module.

~~~
_svtk_module_mark_third_party(<target>)
~~~
#]==]
function (_svtk_module_mark_third_party target)
  # TODO: `_svtk_module_set_module_property` instead.
  set_target_properties("${target}"
    PROPERTIES
      "INTERFACE_svtk_module_exclude_wrap" 1
      "INTERFACE_svtk_module_third_party"  1)
endfunction ()

#[==[
@ingroup module
@brief External third party package

A third party dependency may be expressed as a module using this function.
Third party packages are found using CMake's `find_package` function. It is
highly recommended that imported targets are used to make usage easier. The
module itself will be created as an `INTERFACE` library which exposes the
package.

~~~
svtk_module_third_party_external(
  PACKAGE               <package>
  [VERSION              <version>]
  [COMPONENTS           <component>...]
  [OPTIONAL_COMPONENTS  <component>...]
  [INCLUDE_DIRS <path-or-variable>...]
  [LIBRARIES    <target-or-variable>...]
  [DEFINITIONS  <variable>...]
  [FORWARD_VERSION_REQ  <MAJOR|MINOR|PATCH|EXACT>]
  [VERSION_VAR          <version-spec>]
  [USE_VARIABLES        <variable>...]
  [CONFIG_MODE]
  [STANDARD_INCLUDE_DIRS])
~~~

Only the `PACKAGE` argument is required. The arguments are as follows:

  * `PACKAGE`: (Required) The name of the package to find.
  * `VERSION`: If specified, the minimum version of the dependency that must be
    found.
  * `COMPONENTS`: The list of components to request from the package.
  * `OPTIONAL_COMPONENTS`: The list of optional components to request from the
    package.
  * `STANDARD_INCLUDE_DIRS`: If present, standard include directories will be
    added to the module target. This is usually only required if both internal
    and external are supported for a given dependency.
  * `INCLUDE_DIRS`: If specified, this is added as a `SYSTEM INTERFACE` include
    directory for the target. If a variable name is given, it will be
    dereferenced.
  * `LIBRARIES`: The libraries to link from the package. If a variable name is
    given, it will be dereferenced, however a warning that imported targets are
    not being used will be emitted.
  * `DEFINITIONS`: If specified, the given variables will be added to the
    target compile definitions interface.
  * `CONFIG_MODE`: Force `CONFIG` mode.
  * `FORWARD_VERSION_REQ` and `VERSION_VAR`: See documentation for
    @ref svtk_module_find_package.
  * `USE_VARIABLES`: List of variables from the `find_package` to make
    available to the caller.
#]==]
function (svtk_module_third_party_external)
  cmake_parse_arguments(_svtk_third_party_external
    "STANDARD_INCLUDE_DIRS;CONFIG_MODE"
    "VERSION;PACKAGE;FORWARD_VERSION_REQ;VERSION_VAR"
    "COMPONENTS;OPTIONAL_COMPONENTS;LIBRARIES;INCLUDE_DIRS;DEFINITIONS;TARGETS;USE_VARIABLES"
    ${ARGN})

  if (_svtk_third_party_external_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_third_party_external: "
      "${_svtk_third_party_external_UNPARSED_ARGUMENTS}")
  endif ()

  get_property(_svtk_third_party_external_is_third_party GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_third_party")
  if (NOT _svtk_third_party_external_is_third_party)
    message(FATAL_ERROR
      "The ${_svtk_build_module} has not been declared as a third party "
      "module.")
  endif ()

  if (NOT DEFINED _svtk_third_party_external_PACKAGE)
    message(FATAL_ERROR
      "The `PACKAGE` argument is required.")
  endif ()

  set(_svtk_third_party_external_args)
  if (DEFINED _svtk_third_party_external_FORWARD_VERSION_REQ)
    list(APPEND _svtk_third_party_external_args
      FORWARD_VERSION_REQ "${_svtk_third_party_external_FORWARD_VERSION_REQ}")
  endif ()
  if (DEFINED _svtk_third_party_external_VERSION_VAR)
    list(APPEND _svtk_third_party_external_args
      VERSION_VAR "${_svtk_third_party_external_VERSION_VAR}")
  endif ()

  if (_svtk_third_party_external_TARGETS)
    set(_svtk_third_party_external_config_mode)
    if (_svtk_third_party_external_CONFIG_MODE)
      set(_svtk_third_party_external_config_mode "CONFIG_MODE")
    endif ()

    # If we have targets, they must be exported to the install as well.
    svtk_module_find_package(
      PACKAGE             "${_svtk_third_party_external_PACKAGE}"
      VERSION             "${_svtk_third_party_external_VERSION}"
      COMPONENTS          ${_svtk_third_party_external_COMPONENTS}
      OPTIONAL_COMPONENTS ${_svtk_third_party_external_OPTIONAL_COMPONENTS}
      ${_svtk_third_party_external_config_mode}
      ${_svtk_third_party_external_args})
  else ()
    set(_svtk_third_party_external_config)
    if (_svtk_third_party_external_CONFIG_MODE)
      set(_svtk_third_party_external_config "CONFIG")
    endif ()

    # If there are no targets, the install uses strings and therefore does not
    # need to find the dependency again.
    find_package("${_svtk_third_party_external_PACKAGE}"
      ${_svtk_third_party_external_VERSION}
      ${_svtk_third_party_external_config}
      COMPONENTS          ${_svtk_third_party_external_COMPONENTS}
      OPTIONAL_COMPONENTS ${_svtk_third_party_external_OPTIONAL_COMPONENTS})
  endif ()

  get_property(_svtk_third_party_external_target_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")

  # Check if an imported target of the same name already exists.
  set(_svtk_third_party_external_real_target_name
    "${_svtk_third_party_external_target_name}")
  set(_svtk_third_party_external_using_mangled_name OFF)
  if (TARGET "${_svtk_third_party_external_target_name}")
    # Ensure that the target collision comes from an imported target.
    get_property(_svtk_third_party_external_is_imported
      TARGET    "${_svtk_third_party_external_target_name}"
      PROPERTY  IMPORTED)
    if (NOT _svtk_third_party_external_is_imported)
      message(FATAL_ERROR
        "It appears as though there is a conflicting target named "
        "`${_svtk_third_party_external_target_name}` expected to be used by "
        "the `${_svtk_build_module}` module already added to the build. This "
        "conflicts with the target name expected to be used by an external "
        "third party dependency.")
    endif ()

    # If it does, we need to have a module name that is not the same as this
    # one. Error out if this is detected.
    if (_svtk_build_module STREQUAL _svtk_third_party_external_target_name)
      message(FATAL_ERROR
        "An imported target has the same name used by the module system for "
        "the facade of the external dependency for `${_svtk_build_module}`. "
        "This module must be either renamed or placed into a namespace.")
    endif ()

    # Mangle the internal name. The alias is the expected use case anyways and
    # since this is an INTERFACE target, there's nothing to break with respect
    # to `make $target` anyways.
    string(APPEND _svtk_third_party_external_real_target_name
      "_svtk_module_mangle")
    set_property(GLOBAL APPEND_STRING
      PROPERTY "_svtk_module_${_svtk_build_module}_target_name"
      "_svtk_module_mangle")
    set(_svtk_third_party_external_using_mangled_name ON)
  endif ()

  add_library("${_svtk_third_party_external_real_target_name}" INTERFACE)
  if (_svtk_third_party_external_using_mangled_name)
    set_property(TARGET "${_svtk_third_party_external_real_target_name}"
      PROPERTY
        EXPORT_NAME "${_svtk_third_party_external_target_name}")
  endif ()
  if (NOT _svtk_build_module STREQUAL _svtk_third_party_external_target_name)
    add_library("${_svtk_build_module}" ALIAS
      "${_svtk_third_party_external_real_target_name}")
  endif ()

  if (_svtk_third_party_external_STANDARD_INCLUDE_DIRS)
    _svtk_module_standard_includes(TARGET "${_svtk_third_party_external_real_target_name}"
      SYSTEM INTERFACE)
  endif ()

  # Try to use targets if they're specified and available.
  set(_svtk_third_party_external_have_targets FALSE)
  set(_svtk_third_party_external_used_targets FALSE)
  if (_svtk_third_party_external_TARGETS)
    set(_svtk_third_party_external_have_targets TRUE)
    set(_svtk_third_party_external_all_targets_okay TRUE)
    foreach (_svtk_third_party_external_target IN LISTS _svtk_third_party_external_TARGETS)
      if (NOT TARGET "${_svtk_third_party_external_target}")
        set(_svtk_third_party_external_all_targets_okay FALSE)
        break ()
      endif ()
    endforeach ()

    if (_svtk_third_party_external_all_targets_okay)
      target_link_libraries("${_svtk_third_party_external_real_target_name}"
        INTERFACE
          ${_svtk_third_party_external_TARGETS})
      set(_svtk_third_party_external_used_targets TRUE)
    endif ()
  endif ()

  if (NOT _svtk_third_party_external_used_targets)
    if (NOT _svtk_third_party_external_have_targets)
      message(WARNING
        "A third party dependency for ${_svtk_build_module} was found externally "
        "using paths rather than targets; it is recommended to use imported "
        "targets rather than find_library and such.")
    endif ()

    set(_svtk_third_party_external_have_includes FALSE)
    foreach (_svtk_third_party_external_include_dir IN LISTS _svtk_third_party_external_INCLUDE_DIRS)
      if (DEFINED "${_svtk_third_party_external_include_dir}")
        if (${_svtk_third_party_external_include_dir})
          set(_svtk_third_party_external_have_includes TRUE)
        endif ()
        target_include_directories("${_svtk_third_party_external_real_target_name}" SYSTEM
          INTERFACE "${${_svtk_third_party_external_include_dir}}")
      endif ()
    endforeach ()

    if (_svtk_third_party_external_have_targets AND
        NOT _svtk_third_party_external_have_includes)
      message(WARNING
        "A third party dependency for ${_svtk_build_module} has external targets "
        "which were not found and no `INCLUDE_DIRS` were found either. "
        "Including this module may not work.")
    endif ()

    foreach (_svtk_third_party_external_define IN LISTS _svtk_third_party_external_DEFINITIONS)
      if (DEFINED "${_svtk_third_party_external_define}")
        target_compile_definitions("${_svtk_third_party_external_real_target_name}"
          INTERFACE "${${_svtk_third_party_external_define}}")
      endif ()
    endforeach ()

    set(_svtk_third_party_external_have_libraries FALSE)
    foreach (_svtk_third_party_external_library IN LISTS _svtk_third_party_external_LIBRARIES)
      if (DEFINED "${_svtk_third_party_external_library}")
        if (${_svtk_third_party_external_library})
          set(_svtk_third_party_external_have_libraries TRUE)
        endif ()
        target_link_libraries("${_svtk_third_party_external_real_target_name}"
          INTERFACE "${${_svtk_third_party_external_library}}")
      endif ()
    endforeach ()

    if (_svtk_third_party_external_have_targets AND
        NOT _svtk_third_party_external_have_libraries)
      message(WARNING
        "A third party dependency for ${_svtk_build_module} has external targets "
        "which were not found and no `LIBRARIES` were found either. Linking to "
        "this this module may not work.")
    endif ()
  endif ()

  if (DEFINED _svtk_third_party_external_USE_VARIABLES)
    # If we're called from `svtk_module_third_party`, the variables need bubbled
    # up again.
    if (DEFINED _svtk_third_party_EXTERNAL)
      set(_svtk_third_party_variables
        "${_svtk_third_party_external_USE_VARIABLES}"
        PARENT_SCOPE)
    endif ()

    foreach (_svtk_third_party_external_variable IN LISTS _svtk_third_party_external_USE_VARIABLES)
      if (NOT DEFINED "${_svtk_third_party_external_variable}")
        message(FATAL_ERROR
          "The variable `${_svtk_third_party_external_variable}` was expected "
          "to have been available, but was not defined.")
      endif ()

      set("${_svtk_third_party_external_variable}"
        "${${_svtk_third_party_external_variable}}"
        PARENT_SCOPE)
    endforeach ()
  endif ()

  _svtk_module_mark_third_party("${_svtk_third_party_external_real_target_name}")
  _svtk_module_install("${_svtk_third_party_external_real_target_name}")
endfunction ()

#[==[
@ingroup module
@brief Internal third party package

Third party modules may also be bundled with the project itself. In this case,
it is an internal third party dependency. The dependency is assumed to be in a
subdirectory that will be used via `add_subdirectory`. Unless it is marked as
`HEADERS_ONLY`, it is assumed that it will create a target with the name of the
module.

~~~
svtk_module_third_party_internal(
  [SUBDIRECTORY   <path>]
  [HEADERS_SUBDIR <subdir>]
  [LICENSE_FILES  <file>...]
  [VERSION        <version>]
  [HEADER_ONLY]
  [INTERFACE]
  [STANDARD_INCLUDE_DIRS])
~~~

All arguments are optional, however warnings are emitted if `LICENSE_FILES` or
`VERSION` is not specified. They are as follows:

  * `SUBDIRECTORY`: (Defaults to the library name of the module) The
    subdirectory containing the `CMakeLists.txt` for the dependency.
  * `HEADERS_SUBDIR`: If non-empty, the subdirectory to use for installing
    headers.
  * `LICENSE_FILES`: A list of license files to install for the dependency. If
    not given, a warning will be emitted.
  * `VERSION`: The version of the library that is included.
  * `HEADER_ONLY`: The dependency is header only and will not create a target.
  * `INTERFACE`: The dependency is an `INTERFACE` library.
  * `STANDARD_INCLUDE_DIRS`: If present, module-standard include directories
    will be added to the module target.
#]==]
function (svtk_module_third_party_internal)
  # TODO: Support scanning for third-party modules which don't support an
  # external copy.

  cmake_parse_arguments(_svtk_third_party_internal
    "INTERFACE;HEADER_ONLY;STANDARD_INCLUDE_DIRS"
    "SUBDIRECTORY;HEADERS_SUBDIR;VERSION"
    "LICENSE_FILES"
    ${ARGN})

  if (_svtk_third_party_internal_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_third_party_internal: "
      "${_svtk_third_party_internal_UNPARSED_ARGUMENTS}")
  endif ()

  get_property(_svtk_third_party_internal_is_third_party GLOBAL
    PROPERTY  "_svtk_module_${_svtk_build_module}_third_party")
  if (NOT _svtk_third_party_internal_is_third_party)
    message(FATAL_ERROR
      "The ${_svtk_build_module} has not been declared as a third party "
      "module.")
  endif ()

  get_property(_svtk_third_party_internal_library_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_library_name")
  if (NOT DEFINED _svtk_third_party_internal_SUBDIRECTORY)
    set(_svtk_third_party_internal_SUBDIRECTORY "${_svtk_third_party_internal_library_name}")
  endif ()

  if (NOT DEFINED _svtk_third_party_internal_LICENSE_FILES)
    message(WARNING
      "The ${_svtk_build_module} third party package is embedded, but does not "
      "specify any license files.")
  endif ()

  if (NOT DEFINED _svtk_third_party_internal_VERSION)
    message(WARNING
      "The ${_svtk_build_module} third party package is embedded, but does not "
      "specify the version it is based on.")
  endif ()

  get_property(_svtk_third_party_internal_target_name GLOBAL
    PROPERTY "_svtk_module_${_svtk_build_module}_target_name")
  set(_svtk_third_party_internal_include_type)
  if (_svtk_third_party_internal_INTERFACE)
    set(_svtk_third_party_internal_include_type INTERFACE)
  elseif (_svtk_third_party_internal_HEADER_ONLY)
    add_library("${_svtk_third_party_internal_target_name}" INTERFACE)
    if (NOT _svtk_build_module STREQUAL _svtk_third_party_internal_target_name)
      add_library("${_svtk_build_module}" ALIAS
        "${_svtk_third_party_internal_target_name}")
    endif ()
    set(_svtk_third_party_internal_include_type INTERFACE)
    set(_svtk_third_party_internal_STANDARD_INCLUDE_DIRS 1)
  endif ()

  add_subdirectory("${_svtk_third_party_internal_SUBDIRECTORY}")

  if (NOT TARGET "${_svtk_build_module}")
    message(FATAL_ERROR
      "The ${_svtk_build_module} is being built as an internal third party "
      "library, but a matching target was not created.")
  endif ()

  if (_svtk_third_party_internal_STANDARD_INCLUDE_DIRS)
    _svtk_module_standard_includes(
      TARGET "${_svtk_third_party_internal_target_name}"
      SYSTEM ${_svtk_third_party_internal_include_type}
      HEADERS_DESTINATION "${_svtk_build_HEADERS_DESTINATION}/${_svtk_third_party_internal_HEADERS_SUBDIR}")
  endif ()

  _svtk_module_apply_properties("${_svtk_third_party_internal_target_name}")
  if (_svtk_third_party_internal_INTERFACE)
    # Nothing.
  elseif (_svtk_third_party_internal_HEADER_ONLY)
    _svtk_module_install("${_svtk_third_party_internal_target_name}")
  endif ()

  if (_svtk_third_party_internal_LICENSE_FILES)
    install(
      FILES       ${_svtk_third_party_internal_LICENSE_FILES}
      DESTINATION "${_svtk_build_LICENSE_DESTINATION}/${_svtk_third_party_internal_library_name}/"
      COMPONENT   "license")
  endif ()

  _svtk_module_mark_third_party("${_svtk_third_party_internal_target_name}")
endfunction ()

#[==[
@ingroup module-impl
@brief Output a boolean to JSON

Appends a condition as a JSON boolean with the given dictionary key name to the
given string variable.

~~~
_svtk_json_bool(<output> <name> <cond>)
~~~
#]==]
macro (_svtk_json_bool output name cond)
  if (${cond})
    set(val "true")
  else ()
    set(val "false")
  endif ()
  string(APPEND "${output}" "\"${name}\": ${val}, ")
  unset(val)
endmacro ()

#[==[
@ingroup module-impl
@brief Output a string list to JSON

Appends a variable as a JSON list of strings with the given dictionary key name
to the given string variable.

~~~
_svtk_json_string_list(<output> <name> <cond>)
~~~
#]==]
macro (_svtk_json_string_list output name var)
  set(list "[")
  foreach (value IN LISTS "${var}")
    string(APPEND list "\"${value}\", ")
  endforeach ()
  string(APPEND list "]")
  string(REPLACE ", ]" "]" list "${list}")
  string(APPEND "${output}" "\"${name}\": ${list}, ")
  unset(value)
  unset(list)
endmacro ()

#[==[
@ingroup module-support
@brief JSON metadata representation of modules

Information about the modules built and/or available may be dumped to a JSON
file.

~~~
svtk_module_json(
  MODULES   <module>...
  OUTPUT    <path>)
~~~

  * `MODULES`: (Required) The modules to output information for.
  * `OUTPUT`: (Required) A JSON file describing the modules built will
    be output to this path. Relative paths are rooted to `CMAKE_BINARY_DIR`.

Example output:

~~~{.json}
{
  "modules": [
    {
      "name": "...",
      "library_name": "...",
      "enabled": <bool>,
      "implementable": <bool>,
      "third_party": <bool>,
      "wrap_exclude": <bool>,
      "kit": "...",
      "depends": [
        "..."
      ],
      "optional_depends": [
        "..."
      ],
      "private_depends": [
        "..."
      ],
      "implements": [
        "..."
      ],
      "headers": [
        "..."
      ]
    }
  ],
  "kits": [
    {
      "name": "...",
      "enabled": <bool>,
      "modules": [
      ]
    }
  ]
}
~~~
#]==]
function (svtk_module_json)
  cmake_parse_arguments(_svtk_json
    ""
    "OUTPUT"
    "MODULES"
    ${ARGN})

  if (_svtk_json_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed arguments for svtk_module_json: "
      "${_svtk_json_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_json_OUTPUT)
    message(FATAL_ERROR
      "The `OUTPUT` argument is required.")
  endif ()

  if (NOT _svtk_json_MODULES)
    message(FATAL_ERROR "No modules given to output.")
  endif ()

  if (NOT IS_ABSOLUTE "${_svtk_json_OUTPUT}")
    set(_svtk_json_OUTPUT "${CMAKE_BINARY_DIR}/${_svtk_json_OUTPUT}")
  endif ()

  set(_svtk_json_kits)

  set(_svtk_json_contents "{")
  string(APPEND _svtk_json_contents "\"modules\": {")
  foreach (_svtk_json_module IN LISTS _svtk_json_MODULES)
    get_property(_svtk_json_description GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_description")
    get_property(_svtk_json_implementable GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_implementable")
    get_property(_svtk_json_third_party GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_third_party")
    get_property(_svtk_json_wrap_exclude GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_wrap_exclude")
    get_property(_svtk_json_kit GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_kit")
    get_property(_svtk_json_depends GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_depends")
    get_property(_svtk_json_private_depends GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_private_depends")
    get_property(_svtk_json_optional_depends GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_optional_depends")
    get_property(_svtk_json_implements GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_implements")
    get_property(_svtk_json_library_name GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_library_name")
    get_property(_svtk_json_module_file GLOBAL
      PROPERTY "_svtk_module_${_svtk_json_module}_file")

    set(_svtk_json_kit_name "null")
    if (_svtk_json_kit)
      list(APPEND _svtk_json_kits
        "${_svtk_json_kit}")
      set(_svtk_json_kit_name "\"${_svtk_json_kit}\"")
    endif ()
    set(_svtk_json_headers "")
    if (TARGET "${_svtk_json_module}")
      _svtk_module_get_module_property("${_svtk_json_module}"
        PROPERTY  "headers"
        VARIABLE  _svtk_json_headers)
      get_filename_component(_svtk_json_module_dir "${_svtk_json_module_file}" DIRECTORY)
      file(RELATIVE_PATH _svtk_json_module_subdir "${CMAKE_SOURCE_DIR}" "${_svtk_json_module_dir}")
      string(REPLACE "${CMAKE_SOURCE_DIR}/${_svtk_json_module_subdir}/" "" _svtk_json_headers "${_svtk_json_headers}")
      string(REPLACE "${CMAKE_BINARY_DIR}/${_svtk_json_module_subdir}/" "" _svtk_json_headers "${_svtk_json_headers}")
    endif ()

    string(APPEND _svtk_json_contents "\"${_svtk_json_module}\": {")
    string(APPEND _svtk_json_contents "\"library_name\": \"${_svtk_json_library_name}\", ")
    string(APPEND _svtk_json_contents "\"description\": \"${_svtk_json_description}\", ")
    _svtk_json_bool(_svtk_json_contents "enabled" "TARGET;${_svtk_json_module}")
    _svtk_json_bool(_svtk_json_contents "implementable" _svtk_json_implementable)
    _svtk_json_bool(_svtk_json_contents "third_party" _svtk_json_third_party)
    _svtk_json_bool(_svtk_json_contents "wrap_exclude" _svtk_json_wrap_exclude)
    string(APPEND _svtk_json_contents "\"kit\": ${_svtk_json_kit_name}, ")
    _svtk_json_string_list(_svtk_json_contents "depends" _svtk_json_depends)
    _svtk_json_string_list(_svtk_json_contents "optional_depends" _svtk_json_optional_depends)
    _svtk_json_string_list(_svtk_json_contents "private_depends" _svtk_json_private_depends)
    _svtk_json_string_list(_svtk_json_contents "implements" _svtk_json_implements)
    _svtk_json_string_list(_svtk_json_contents "headers" _svtk_json_headers)
    string(APPEND _svtk_json_contents "}, ")
  endforeach ()
  string(APPEND _svtk_json_contents "}, ")

  string(APPEND _svtk_json_contents "\"kits\": {")
  foreach (_svtk_json_kit IN LISTS _svtk_json_kits)
    set(_svtk_json_library_name "null")
    if (TARGET "${_svtk_json_kit}")
      get_property(_svtk_json_library
        TARGET    "${_svtk_json_kit}"
        PROPERTY  LIBRARY_OUTPUT_NAME)
      set(_svtk_json_library_name "\"${_svtk_json_library}\"")
    endif ()

    string(APPEND _svtk_json_contents "\"${_svtk_json_kit}\": {")
    string(APPEND _svtk_json_contents "\"library_name\": ${_svtk_json_library_name}, ")
    _svtk_json_bool(_svtk_json_contents "enabled" "TARGET;${_svtk_json_kit}")
    string(APPEND _svtk_json_contents "}, ")
  endforeach ()
  string(APPEND _svtk_json_contents "}, ")

  string(APPEND _svtk_json_contents "}")
  string(REPLACE ", ]" "]" _svtk_json_contents "${_svtk_json_contents}")
  string(REPLACE ", }" "}" _svtk_json_contents "${_svtk_json_contents}")
  file(GENERATE
    OUTPUT  "${_svtk_json_OUTPUT}"
    CONTENT "${_svtk_json_contents}")
endfunction ()

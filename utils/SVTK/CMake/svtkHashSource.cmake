#[==[
@file svtkHashSource.cmake

This module contains the @ref svtk_hash_source function which may be used to
generate a hash from a file and place that in a generated header.
#]==]

set(_svtkHashSource_script_file "${CMAKE_CURRENT_LIST_FILE}")

include(CMakeParseArguments)

#[==[
@brief Generate a header containing the hash of a file

Add a rule to turn a file into a MD5 hash and place that in a C string.

~~~
svtk_hash_source(
  INPUT          <input>
  [NAME          <name>]
  [ALGORITHM     <algorithm>]
  [HEADER_OUTPUT <header>])
~~~

The only required variable is `INPUT`.

  * `INPUT`: (Required) The path to the file to process. If a relative path
    is given, it will be interpreted as being relative to
    `CMAKE_CURRENT_SOURCE_DIR`.
  * `NAME`: This is the base name of the header file that will be generated as
    well as the variable name for the C string. It defaults to basename of the
    input suffixed with `Hash`.
  * `ALGORITHM`: This is the hashing algorithm to use. Supported values are
    MD5, SHA1, SHA224, SHA256, SHA384, and SHA512. If not specified, MD5 is assumed.
  * `HEADER_OUTPUT`: the variable to store the generated header path.
#]==]
function (svtk_hash_source)
  cmake_parse_arguments(_svtk_hash_source
    ""
    "INPUT;NAME;ALGORITHM;HEADER_OUTPUT"
    ""
    ${ARGN})

  if (_svtk_hash_source_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unrecognized arguments to svtk_hash_source: "
      "${_svtk_hash_source_UNPARSED_ARGUMENTS}")
  endif ()

  if (NOT DEFINED _svtk_hash_source_INPUT)
    message(FATAL_ERROR
      "Missing `INPUT` for svtk_hash_source.")
  endif ()

  if (NOT DEFINED _svtk_hash_source_NAME)
    get_filename_component(_svtk_hash_source_NAME
      "${_svtk_hash_source_INPUT}" NAME_WE)
    set(_svtk_hash_source_NAME "${_svtk_hash_source_NAME}Hash")
  endif ()

  if (NOT DEFINED _svtk_hash_source_ALGORITHM)
    set(_svtk_hash_source_ALGORITHM MD5)
  endif ()

  if (IS_ABSOLUTE "${_svtk_hash_source_INPUT}")
    set(_svtk_hash_source_input
      "${_svtk_hash_source_INPUT}")
  else ()
    set(_svtk_hash_source_input
      "${CMAKE_CURRENT_SOURCE_DIR}/${_svtk_hash_source_INPUT}")
  endif ()

  set(_svtk_hash_source_header
    "${CMAKE_CURRENT_BINARY_DIR}/${_svtk_hash_source_NAME}.h")

  add_custom_command(
    OUTPUT  "${_svtk_hash_source_header}"
    DEPENDS "${_svtkHashSource_script_file}"
            "${_svtk_hash_source_input}"
    COMMAND "${CMAKE_COMMAND}"
            "-Dinput_file=${_svtk_hash_source_input}"
            "-Doutput_file=${_svtk_hash_source_header}"
            "-Doutput_name=${_svtk_hash_source_NAME}"
            "-Dalgorithm=${_svtk_hash_source_ALGORITHM}"
            "-D_svtk_hash_source_run=ON"
            -P "${_svtkHashSource_script_file}")

  if (DEFINED _svtk_hash_source_HEADER_OUTPUT)
    set("${_svtk_hash_source_HEADER_OUTPUT}"
      "${_svtk_hash_source_header}"
      PARENT_SCOPE)
  endif ()
endfunction()

if (_svtk_hash_source_run AND CMAKE_SCRIPT_MODE_FILE)
  file(${algorithm} "${input_file}" file_hash)
  file(WRITE "${output_file}"
    "#ifndef ${output_name}\n #define ${output_name} \"${file_hash}\"\n#endif\n")
endif ()

function (svtk_type_native type ctype class)
  string(TOUPPER "${type}" type_upper)
  set("svtk_type_native_${type}" "
#if SVTK_TYPE_${type_upper} == SVTK_${ctype}
# include \"${class}Array.h\"
# define svtkTypeArrayBase ${class}Array
#endif
"
    PARENT_SCOPE)
endfunction ()

function (svtk_type_native_fallback type preferred_ctype preferred_class fallback_class)
  string(TOUPPER "${type}" type_upper)
  set("svtk_type_native_${type}" "
#if SVTK_TYPE_${type_upper} == SVTK_${preferred_ctype}
# include \"${preferred_class}Array.h\"
# define svtkTypeArrayBase ${preferred_class}Array
#else
# include \"${fallback_class}Array.h\"
# define svtkTypeArrayBase ${fallback_class}Array
#endif
"
    PARENT_SCOPE)
endfunction ()

function (svtk_type_native_choice type preferred_ctype preferred_class fallback_ctype fallback_class)
  string(TOUPPER "${type}" type_upper)
  set("svtk_type_native_${type}" "
#if SVTK_TYPE_${type_upper} == SVTK_${preferred_ctype}
# include \"${preferred_class}Array.h\"
# define svtkTypeArrayBase ${preferred_class}Array
#elif SVTK_TYPE_${type_upper} == SVTK_${fallback_ctype}
# include \"${fallback_class}Array.h\"
# define svtkTypeArrayBase ${fallback_class}Array
#endif
"
    PARENT_SCOPE)
endfunction ()

# Configure data arrays for platform-independent fixed-size types.
# Match the type selection here to that in svtkType.h.
svtk_type_native_fallback(Int8 CHAR svtkChar svtkSignedChar)
svtk_type_native(UInt8 UNSIGNED_CHAR svtkUnsignedChar)
svtk_type_native(Int16 SHORT svtkShort)
svtk_type_native(UInt16 UNSIGNED_SHORT svtkUnsignedShort)
svtk_type_native(Int32 INT svtkInt)
svtk_type_native(UInt32 UNSIGNED_INT svtkUnsignedInt)
svtk_type_native_choice(Int64 LONG svtkLong LONG_LONG svtkLongLong)
svtk_type_native_choice(UInt64 UNSIGNED_LONG svtkUnsignedLong UNSIGNED_LONG_LONG svtkUnsignedLongLong)
svtk_type_native(Float32 FLOAT svtkFloat)
svtk_type_native(Float64 DOUBLE svtkDouble)

foreach(svtk_type IN ITEMS Int8 Int16 Int32 Int64 UInt8 UInt16 UInt32 UInt64 Float32 Float64)
  set(SVTK_TYPE_NAME ${svtk_type})
  set(SVTK_TYPE_NATIVE "${svtk_type_native_${svtk_type}}")
  if(SVTK_TYPE_NATIVE)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/svtkTypedArray.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/svtkType${svtk_type}Array.h"
      @ONLY)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/svtkTypedArray.cxx.in"
      "${CMAKE_CURRENT_BINARY_DIR}/svtkType${svtk_type}Array.cxx"
      @ONLY)
    list(APPEND sources
      "${CMAKE_CURRENT_BINARY_DIR}/svtkType${svtk_type}Array.cxx")
    list(APPEND headers
      "${CMAKE_CURRENT_BINARY_DIR}/svtkType${svtk_type}Array.h")
  endif()
endforeach()

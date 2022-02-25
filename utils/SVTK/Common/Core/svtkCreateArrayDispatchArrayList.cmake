# This file contains macros that are used by SVTK to generate the list of
# default arrays used by the svtkArrayDispatch system.
#
# There are a number of CMake variables that control the final array list. At
# the high level, the following options enable/disable predefined categories of
# arrays:
#
# - SVTK_DISPATCH_AOS_ARRAYS (default: ON)
#   Include svtkAOSDataArrayTemplate<ValueType> for the basic types supported
#   by SVTK. This should probably not be turned off.
# - SVTK_DISPATCH_SOA_ARRAYS (default: OFF)
#   Include svtkSOADataArrayTemplate<ValueType> for the basic types supported
#   by SVTK.
# - SVTK_DISPATCH_TYPED_ARRAYS (default: OFF)
#   Include svtkTypedDataArray<ValueType> for the basic types supported
#   by SVTK. This enables the old-style in-situ svtkMappedDataArray subclasses
#   to be used.
#
# At a lower level, specific arrays can be added to the list individually in
# two ways:
#
# For templated classes, set the following variables:
# - svtkArrayDispatch_containers:
#   List of template class names.
# - svtkArrayDispatch_[template class name]_types:
#   For the specified template class, add an entry to the array list that
#   instantiates the container for each type listed here.
# - svtkArrayDispatch_[template class name]_header
#   Specifies the header file to include for the specified template class.
#
# Both templated and non-templated arrays can be added using these variables:
# - svtkArrayDispatch_extra_arrays:
#   List of arrays to add to the list.
# - svtkArrayDispatch_extra_headers:
#   List of headers to include.
#
################################ Example #######################################
#
# The cmake call below instantiates the array list that follows:
#
# cmake [path to SVTK source]
#   -DsvtkArrayDispatch_containers="MyCustomArray1;MyCustomArray2"
#   -DsvtkArrayDispatch_MyCustomArray1_header="MyCustomArray1.h"
#   -DsvtkArrayDispatch_MyCustomArray1_types="float;double"
#   -DsvtkArrayDispatch_MyCustomArray2_header="MyCustomArray2.h"
#   -DsvtkArrayDispatch_MyCustomArray2_types="int;unsigned char"
#   -DsvtkArrayDispatch_extra_headers="ExtraHeader1.h;ExtraHeader2.h"
#   -DsvtkArrayDispatch_extra_arrays="ExtraArray1;ExtraArray2<float>;ExtraArray2<char>"
#
# Generated header:
#
# #ifndef svtkArrayDispatchArrayList_h
# #define svtkArrayDispatchArrayList_h
#
# #include "svtkTypeList.h"
# #include "MyCustomArray1.h"
# #include "MyCustomArray2.h"
# #include "svtkAOSDataArrayTemplate.h"
# #include "ExtraHeader1.h"
# #include "ExtraHeader2.h"
#
# namespace svtkArrayDispatch {
#
# typedef svtkTypeList::Unique<
#   svtkTypeList::Create<
#     MyCustomArray1<float>,
#     MyCustomArray1<double>,
#     MyCustomArray2<int>,
#     MyCustomArray2<unsigned char>,
#     svtkAOSDataArrayTemplate<char>,
#     svtkAOSDataArrayTemplate<double>,
#     svtkAOSDataArrayTemplate<float>,
#     svtkAOSDataArrayTemplate<int>,
#     svtkAOSDataArrayTemplate<long>,
#     svtkAOSDataArrayTemplate<short>,
#     svtkAOSDataArrayTemplate<signed char>,
#     svtkAOSDataArrayTemplate<unsigned char>,
#     svtkAOSDataArrayTemplate<unsigned int>,
#     svtkAOSDataArrayTemplate<unsigned long>,
#     svtkAOSDataArrayTemplate<unsigned short>,
#     svtkAOSDataArrayTemplate<svtkIdType>,
#     svtkAOSDataArrayTemplate<long long>,
#     svtkAOSDataArrayTemplate<unsigned long long>,
#     ExtraArray1,
#     ExtraArray2<float>,
#     ExtraArray2<char>
#   >
# >::Result Arrays;
#
# } // end namespace svtkArrayDispatch
#
# #endif // svtkArrayDispatchArrayList_h
#

# Populate the environment so that svtk_array_dispatch_generate_array_header will
# create the array TypeList with all known array types.
macro(svtkArrayDispatch_default_array_setup)

# The default set of scalar types:
set(svtkArrayDispatch_all_types
  "char"
  "double"
  "float"
  "int"
  "long"
  "long long"
  "short"
  "signed char"
  "unsigned char"
  "unsigned int"
  "unsigned long"
  "unsigned long long"
  "unsigned short"
  "svtkIdType"
)

# For each container, define a header and a list of types:
if (SVTK_DISPATCH_AOS_ARRAYS)
  list(APPEND svtkArrayDispatch_containers svtkAOSDataArrayTemplate)
  set(svtkArrayDispatch_svtkAOSDataArrayTemplate_header svtkAOSDataArrayTemplate.h)
  set(svtkArrayDispatch_svtkAOSDataArrayTemplate_types
    ${svtkArrayDispatch_all_types}
  )
endif()

if (SVTK_DISPATCH_SOA_ARRAYS)
  list(APPEND svtkArrayDispatch_containers svtkSOADataArrayTemplate)
  set(svtkArrayDispatch_svtkSOADataArrayTemplate_header svtkSOADataArrayTemplate.h)
  set(svtkArrayDispatch_svtkSOADataArrayTemplate_types
    ${svtkArrayDispatch_all_types}
  )
  if (SVTK_BUILD_SCALED_SOA_ARRAYS)
    list(APPEND svtkArrayDispatch_containers svtkScaledSOADataArrayTemplate)
    set(svtkArrayDispatch_svtkScaledSOADataArrayTemplate_header svtkScaledSOADataArrayTemplate.h)
    set(svtkArrayDispatch_svtkScaledSOADataArrayTemplate_types
      ${svtkArrayDispatch_all_types}
    )
  endif()
endif()

if (SVTK_DISPATCH_TYPED_ARRAYS)
  list(APPEND svtkArrayDispatch_containers svtkTypedDataArray)
  set(svtkArrayDispatch_svtkTypedDataArray_header svtkTypedDataArray.h)
  set(svtkArrayDispatch_svtkTypedDataArray_types
    ${svtkArrayDispatch_all_types}
  )
endif()

endmacro()

# Concatenates a list of strings into a single string, since string(CONCAT ...)
# is not currently available for SVTK's cmake version.
# Internal method.
function(CollapseString input output)
  set(temp "")
  foreach(line ${input})
    set(temp ${temp}${line})
  endforeach()
  set(${output} "${temp}" PARENT_SCOPE)
endfunction()

# Create a header that declares the svtkArrayDispatch::Arrays TypeList.
macro(svtkArrayDispatch_generate_array_header result)

set(svtkAD_headers svtkTypeList.h)
set(svtkAD_arrays)
foreach(container ${svtkArrayDispatch_containers})
  list(APPEND svtkAD_headers ${svtkArrayDispatch_${container}_header})
  foreach(value_type ${svtkArrayDispatch_${container}_types})
    list(APPEND svtkAD_arrays "${container}<${value_type}>")
  endforeach()
endforeach()

# Include externally specified headers/arrays:
list(APPEND svtkAD_headers ${svtkArrayDispatch_extra_headers})
list(APPEND svtkAD_arrays ${svtkArrayDispatch_extra_arrays})

set(temp
  "// This file is autogenerated by svtkCreateArrayDispatchArrayList.cmake.\n"
  "// Do not edit this file. Your changes will not be saved.\n"
  "\n"
  "#ifndef svtkArrayDispatchArrayList_h\n"
  "#define svtkArrayDispatchArrayList_h\n"
  "\n"
)

foreach(header ${svtkAD_headers})
  list(APPEND temp "#include \"${header}\"\n")
endforeach()

list(APPEND temp
  "\n"
  "namespace svtkArrayDispatch {\n"
  "\n"
  "typedef svtkTypeList::Unique<\n"
  "  svtkTypeList::Create<\n"
)

foreach(array ${svtkAD_arrays})
  list(APPEND temp "    ${array},\n")
endforeach()

# Remove the final comma from the array list:
CollapseString("${temp}" temp)
string(REGEX REPLACE ",\n$" "\n" temp "${temp}")

list(APPEND temp
  "  >\n"
  ">::Result Arrays\;\n"
  "\n"
  "} // end namespace svtkArrayDispatch\n"
  "\n"
  "#endif // svtkArrayDispatchArrayList_h\n"
)

CollapseString("${temp}" ${result})

endmacro()

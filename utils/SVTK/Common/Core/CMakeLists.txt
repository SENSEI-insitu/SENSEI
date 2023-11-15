# Core configuration options.
if(CMAKE_SIZEOF_VOID_P GREATER 4)
 set(SVTK_USE_64BIT_IDS_DEFAULT ON)
else()
 set(SVTK_USE_64BIT_IDS_DEFAULT OFF)
 set(SVTK_USE_64BIT_TIMESTAMPS_DEFAULT OFF)
 option(SVTK_USE_64BIT_TIMESTAMPS "Build SVTK with 64 bit timestamps"
  ${SVTK_USE_64BIT_TIMESTAMPS_DEFAULT})
endif()

option(SVTK_USE_64BIT_IDS "Build SVTK with 64 bit ids"
  ${SVTK_USE_64BIT_IDS_DEFAULT})
mark_as_advanced(SVTK_USE_64BIT_IDS)

option(SVTK_DEBUG_LEAKS "Build leak checking support into SVTK." OFF)
mark_as_advanced(SVTK_DEBUG_LEAKS)
include(CMakeDependentOption)
# See svtkDataArrayRange.h docs for more info on these:
cmake_dependent_option(SVTK_DEBUG_RANGE_ITERATORS
  "Detect usage errors with the for-range iterators (SLOW)." OFF
  "CMAKE_BUILD_TYPE STREQUAL \"Debug\"" OFF)
mark_as_advanced(SVTK_DEBUG_RANGE_ITERATORS)
cmake_dependent_option(SVTK_ALWAYS_OPTIMIZE_ARRAY_ITERATORS
  "Enable additional optimizations for array iterators in Debug builds." OFF
  "NOT SVTK_DEBUG_RANGE_ITERATORS" OFF)
mark_as_advanced(SVTK_ALWAYS_OPTIMIZE_ARRAY_ITERATORS)

# This option determines the behavior of the New methods of svtkObject derived
# classes. If this option is off then only classes specifically using the
# svtkObjectFactoryNewMacro and svtkAbstractObjectFactoryNewMacro will allow for
# overrides. If the method is on then svtkStandardNewMacro will use the
# svtkObjectFactoryNewMacro's implementation body.
option(SVTK_ALL_NEW_OBJECT_FACTORY
  "Build all svtkObject derived classes with object factory new methods." OFF)
mark_as_advanced(SVTK_ALL_NEW_OBJECT_FACTORY)

set(svtk_smp_sources)
set(svtk_smp_headers)
set(svtk_smp_defines)
set(svtk_smp_libraries)
include("${CMAKE_CURRENT_SOURCE_DIR}/svtkSMPSelection.cmake")

# Generate the svtkTypeList_Create macros:
include("${CMAKE_CURRENT_SOURCE_DIR}/svtkCreateTypeListMacros.cmake")
CreateTypeListMacros(
  SVTK_TYPELISTMACRO_HEADER_CONTENTS
  99
  svtkTypeList_Create
  "svtkTypeList::TypeList"
  "svtkTypeList::NullType")
configure_file(svtkTypeListMacros.h.in svtkTypeListMacros.h)

# And the ArrayDispatch array list header:
option(SVTK_DISPATCH_AOS_ARRAYS "Include array-of-structs svtkDataArray subclasses in dispatcher." ON)
option(SVTK_DISPATCH_SOA_ARRAYS "Include struct-of-arrays svtkDataArray subclasses in dispatcher." OFF)
option(SVTK_DISPATCH_TYPED_ARRAYS "Include svtkTypedDataArray subclasses (e.g. old mapped arrays) in dispatcher." OFF)
option(SVTK_WARN_ON_DISPATCH_FAILURE "If enabled, svtkArrayDispatch will print a warning when a dispatch fails." OFF)
mark_as_advanced(
  SVTK_DISPATCH_AOS_ARRAYS
  SVTK_DISPATCH_SOA_ARRAYS
  SVTK_DISPATCH_TYPED_ARRAYS
  SVTK_WARN_ON_DISPATCH_FAILURE)

option(SVTK_BUILD_SCALED_SOA_ARRAYS "Include struct-of-arrays with scaled svtkDataArray implementation." OFF)

include("${CMAKE_CURRENT_SOURCE_DIR}/svtkCreateArrayDispatchArrayList.cmake")
svtkArrayDispatch_default_array_setup()
svtkArrayDispatch_generate_array_header(SVTK_ARRAYDISPATCH_ARRAY_LIST)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkArrayDispatchArrayList.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkArrayDispatchArrayList.h"
  @ONLY)

set(classes
  svtkAbstractArray
  svtkAnimationCue
  svtkArchiver
  svtkArray
  svtkArrayCoordinates
  svtkArrayExtents
  svtkArrayExtentsList
  svtkArrayIterator
  svtkArrayRange
  svtkArraySort
  svtkArrayWeights
  svtkBitArray
  svtkBitArrayIterator
  svtkBoxMuellerRandomSequence
  svtkBreakPoint
  svtkByteSwap
  svtkCallbackCommand
  svtkCharArray
  svtkCollection
  svtkCollectionIterator
  svtkCommand
  svtkCommonInformationKeyManager
  svtkConditionVariable
  svtkCriticalSection
  svtkDataArray
  svtkDataArrayCollection
  svtkDataArrayCollectionIterator
  svtkDataArraySelection
  svtkDebugLeaks
  svtkDebugLeaksManager
  svtkDoubleArray
  svtkDynamicLoader
  svtkEventForwarderCommand
  svtkFileOutputWindow
  svtkFloatArray
  svtkFloatingPointExceptions
  svtkGarbageCollector
  svtkGarbageCollectorManager
  svtkGaussianRandomSequence
  svtkIdList
  svtkIdListCollection
  svtkIdTypeArray
  svtkIndent
  svtkInformation
  svtkInformationDataObjectKey
  svtkInformationDoubleKey
  svtkInformationDoubleVectorKey
  svtkInformationIdTypeKey
  svtkInformationInformationKey
  svtkInformationInformationVectorKey
  svtkInformationIntegerKey
  svtkInformationIntegerPointerKey
  svtkInformationIntegerVectorKey
  svtkInformationIterator
  svtkInformationKey
  svtkInformationKeyLookup
  svtkInformationKeyVectorKey
  svtkInformationObjectBaseKey
  svtkInformationObjectBaseVectorKey
  svtkInformationRequestKey
  svtkInformationStringKey
  svtkInformationStringVectorKey
  svtkInformationUnsignedLongKey
  svtkInformationVariantKey
  svtkInformationVariantVectorKey
  svtkInformationVector
  svtkIntArray
  svtkLargeInteger
  svtkLogger
  svtkLongArray
  svtkLongLongArray
  svtkLookupTable
  svtkMath
  svtkMersenneTwister
  svtkMinimalStandardRandomSequence
  svtkMultiThreader
  svtkMutexLock
  svtkOStrStreamWrapper
  svtkOStreamWrapper
  svtkObject
  svtkObjectBase
  svtkObjectFactory
  svtkObjectFactoryCollection
  svtkOldStyleCallbackCommand
  svtkOutputWindow
  svtkOverrideInformation
  svtkOverrideInformationCollection
  svtkPoints
  svtkPoints2D
  svtkPriorityQueue
  svtkRandomPool
  svtkRandomSequence
  svtkReferenceCount
  svtkScalarsToColors
  svtkShortArray
  svtkSignedCharArray
  svtkSimpleCriticalSection
  svtkSmartPointerBase
  svtkSortDataArray
  svtkStdString
  svtkStringArray
  svtkStringOutputWindow
  svtkTimePointUtility
  svtkTimeStamp
  svtkUnicodeString
  svtkUnicodeStringArray
  svtkUnsignedCharArray
  svtkUnsignedIntArray
  svtkUnsignedLongArray
  svtkUnsignedLongLongArray
  svtkUnsignedShortArray
  svtkVariant
  svtkVariantArray
  svtkVersion
  svtkVoidArray
  svtkWeakPointerBase
  svtkWeakReference
  svtkWindow
  svtkXMLFileOutputWindow)

set(template_classes
  svtkAOSDataArrayTemplate
  svtkArrayDispatch
  svtkArrayInterpolate
  svtkArrayIteratorTemplate
  svtkArrayPrint
  svtkDenseArray
  svtkGenericDataArray
  svtkMappedDataArray
  svtkSOADataArrayTemplate
  svtkSparseArray
  svtkTypedArray
  svtkTypedDataArray
  svtkTypeList)

set(sources
  svtkArrayIteratorTemplateInstantiate.cxx
  svtkGenericDataArray.cxx
  svtkSOADataArrayTemplateInstantiate.cxx
  ${svtk_smp_sources})

if (SVTK_BUILD_SCALED_SOA_ARRAYS)
  list (APPEND template_classes svtkScaledSOADataArrayTemplate)
  list (APPEND sources  svtkScaledSOADataArrayTemplateInstantiate.cxx)
  set(SVTK_USE_SCALED_SOA_ARRAYS 1)
endif ()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkSVTK_USE_SCALED_SOA_ARRAYS.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkSVTK_USE_SCALED_SOA_ARRAYS.h")

set(headers
  svtkABI.h
  svtkArrayIteratorIncludes.h
  svtkAssume.h
  svtkAtomicTypeConcepts.h
  svtkAutoInit.h
  svtkBuffer.h
  svtkCollectionRange.h
  svtkDataArrayAccessor.h
  svtkDataArrayIteratorMacro.h
  svtkDataArrayMeta.h
  svtkDataArrayRange.h
  svtkDataArrayTemplate.h
  svtkDataArrayTupleRange_AOS.h
  svtkDataArrayTupleRange_Generic.h
  svtkDataArrayValueRange_AOS.h
  svtkDataArrayValueRange_Generic.h
  svtkEventData.h
  svtkGenericDataArrayLookupHelper.h
  svtkIOStream.h
  svtkIOStreamFwd.h
  svtkInformationInternals.h
  svtkIterator.h
  svtkMathUtilities.h
  svtkMeta.h
  svtkNew.h
  svtkRange.h
  svtkRangeIterableTraits.h
  svtkSetGet.h
  svtkSmartPointer.h
  svtkSystemIncludes.h
  svtkTemplateAliasMacro.h
  svtkTestDataArray.h
  svtkType.h
  svtkTypeTraits.h
  svtkTypedDataArrayIterator.h
  svtkVariantCast.h
  svtkVariantCreate.h
  svtkVariantExtract.h
  svtkVariantInlineOperators.h
  svtkWeakPointer.h
  svtkWin32Header.h
  svtkWindows.h
  svtkWrappingHints.h
  ${svtk_smp_headers}
  "${CMAKE_CURRENT_BINARY_DIR}/svtkArrayDispatchArrayList.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkConfigure.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkMathConfigure.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkToolkits.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkTypeListMacros.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkVersionMacros.h"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkSVTK_USE_SCALED_SOA_ARRAYS.h")

set(private_headers
  "${CMAKE_CURRENT_BINARY_DIR}/svtkFloatingPointExceptionsConfigure.h")

set(templates
  svtkArrayIteratorTemplateImplicit.txx)

set(private_templates
  svtkDataArrayPrivate.txx)

set(svtk_include_dirs)

svtk_module_find_package(
  PACKAGE Threads)

set(SVTK_MAX_THREADS "64" CACHE STRING
  "Max number of threads svtkMultiThreader will allocate.")
mark_as_advanced(SVTK_MAX_THREADS)

if (CMAKE_USE_PTHREADS_INIT AND CMAKE_USE_WIN32_THREADS_INIT)
  if (DEFINED _svtk_thread_impl_output)
    set(thread_impl_output "${_svtk_thread_impl_output}")
  else ()
    execute_process(
      COMMAND "${CMAKE_C_COMPILER}" -v
      OUTPUT_VARIABLE thread_impl_output
      ERROR_VARIABLE  thread_impl_output
      RESULT_VARIABLE result
      TIMEOUT         10)
    if (result)
      message(FATAL_ERROR
        "Failed to determine the thread implementation in use by the "
        "compiler.")
    endif ()
    string (REGEX REPLACE ".*Thread model: \((posix|win32)\).*" "\\1" thread_impl_output "${thread_impl_output}")
  endif ()
  set(_svtk_thread_impl_output "${thread_impl_output}"
    CACHE INTERNAL "thread model detection")
  if (thread_impl_output MATCHES "posix")
    set(SVTK_USE_PTHREADS 1)
  elseif (thread_impl_output MATCHES "win32")
    set(SVTK_USE_WIN32_THREADS 1)
  else ()
    set(SVTK_USE_PTHREADS 1)
  endif ()
elseif (CMAKE_USE_PTHREADS_INIT)
  set(SVTK_USE_PTHREADS 1)
elseif (CMAKE_USE_WIN32_THREADS_INIT)
  set(SVTK_USE_WIN32_THREADS 1)
endif ()

# We can't check for the target since we're before SVTK::CommonDataModel in the
# dependency graph.
if (SVTK_MODULE_ENABLE_SVTK_CommonDataModel)
  list(APPEND svtk_include_dirs
    "${CMAKE_CURRENT_BINARY_DIR}/../DataModel")
  set_property(SOURCE svtkInformationDataObjectKey.cxx
    PROPERTY
      COMPILE_DEFINITIONS svtkCommonDataModel_ENABLED)
endif ()

if(WIN32)
  list(APPEND classes
    svtkWin32OutputWindow
    svtkWin32ProcessOutputWindow)
  list(APPEND sources
    svtkWin32OutputWindowProcessEncoded.c)
endif()

if(ANDROID)
  list(APPEND classes
    svtkAndroidOutputWindow)
endif()

# Platform configuration tests.
include(TestBigEndian)
test_big_endian(SVTK_WORDS_BIGENDIAN)

# Provide version macros through svtkConfigure.h for compatibility.
if(SVTK_LEGACY_REMOVE)
  set(SVTK_CONFIG_LEGACY_VERSION "")
else()
  set(SVTK_CONFIG_LEGACY_VERSION "#include \"svtkVersionMacros.h\" // removed by SVTK_LEGACY_REMOVE")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkVersionMacros.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkVersionMacros.h"
  @ONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkConfigure.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkConfigure.h"
  @ONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkToolkits.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkToolkits.h"
  @ONLY)

add_custom_command(
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/CaseFolding.txt"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/svtkUnicodeString.cmake"
  COMMAND "${CMAKE_COMMAND}"
          "-Dsource_dir=${CMAKE_CURRENT_SOURCE_DIR}"
          "-Dbinary_dir=${CMAKE_CURRENT_BINARY_DIR}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/svtkUnicodeString.cmake"
  OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/svtkUnicodeCaseFoldData.h")

set_source_files_properties(svtkUnicodeString.cxx
  PROPERTIES
    OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/svtkUnicodeCaseFoldData.h")

include("${CMAKE_CURRENT_SOURCE_DIR}/svtkTypeArrays.cmake")

include(CheckCXXSourceCompiles)

macro(CHECK_CXX_EXPRESSION_COMPILES EXPRESSION FILES VARIABLE)
  set(SOURCE "/* CHECK_CXX_EXPRESSION_COMPILES */\n")
  foreach(FILE ${FILES})
    set(SOURCE "${SOURCE}#include <${FILE}>\n")
  endforeach()
  set(SOURCE "${SOURCE}\nint main()\n{\n")
  set(SOURCE "${SOURCE}  static_cast<void>(${EXPRESSION});\n\n")
  set(SOURCE "${SOURCE}  return 0;\n}\n")
  check_cxx_source_compiles("${SOURCE}" "${VARIABLE}")
endmacro()

# Check C++ <cmath> first, where the C++11 standard says these must be.
check_cxx_expression_compiles("std::isnan(0.0)" "cmath" SVTK_HAS_STD_ISNAN)
check_cxx_expression_compiles("std::isinf(0.0)" "cmath" SVTK_HAS_STD_ISINF)
check_cxx_expression_compiles("std::isfinite(0.0)" "cmath" SVTK_HAS_STD_ISFINITE)

include(CheckSymbolExists)

# Check C99 <math.h> next, where the C99 standard says these must be.
# (they will be found even if they are defined as macros)
check_symbol_exists(isnan "math.h" SVTK_HAS_ISNAN)
check_symbol_exists(isinf "math.h" SVTK_HAS_ISINF)
check_symbol_exists(isfinite "math.h" SVTK_HAS_ISFINITE)

# Check for finite() which is widespread, though non-standard.
check_symbol_exists(finite "math.h" SVTK_HAS_FINITE)

# Check float.h last, where Windows keeps its underscore versions.
if(NOT SVTK_HAS_ISNAN AND NOT SVTK_HAS_STD_ISNAN)
  check_symbol_exists(_isnan "float.h" SVTK_HAS__ISNAN)
endif()
if(NOT SVTK_HAS_FINITE AND NOT SVTK_HAS_ISFINITE AND NOT SVTK_HAS_STD_ISFINITE)
  check_symbol_exists(_finite "float.h" SVTK_HAS__FINITE)
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkMathConfigure.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkMathConfigure.h")

# Figure out if feenableexcept and fedisableexcept functions are present
include(CheckIncludeFile)
check_include_file("fenv.h" HAVE_FENV_H)
if(HAVE_FENV_H)
  set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
  if(UNIX)
    set(CMAKE_REQUIRED_LIBRARIES m)
  endif()
  check_symbol_exists(feenableexcept "fenv.h" SVTK_HAS_FEENABLEEXCEPT)
  set(CMAKE_REQUIRED_LIBRARIES)
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/svtkFloatingPointExceptionsConfigure.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/svtkFloatingPointExceptionsConfigure.h")

if(MSVC)
  set_source_files_properties(
    svtkDataArray.cxx
    PROPERTIES
      COMPILE_FLAGS /bigobj)
endif()

svtk_module_add_module(SVTK::CommonCore
  CLASSES           ${classes}
  TEMPLATE_CLASSES  ${template_classes}
  SOURCES           ${sources}
  TEMPLATES         ${templates}
  HEADERS           ${headers}
  PRIVATE_HEADERS   ${private_headers}
  PRIVATE_TEMPLATES ${private_templates})
if (svtk_include_dirs)
  svtk_module_include(SVTK::CommonCore
    PRIVATE
      ${svtk_include_dirs})
endif ()

svtk_module_link(SVTK::CommonCore
  PUBLIC
    Threads::Threads
    ${svtk_smp_libraries}
  PRIVATE
    # Need nsl to resolve gethostbyname on SunOS-5.8 and socket also
    $<$<PLATFORM_ID:SunOS>:socket>
    $<$<PLATFORM_ID:SunOS>:nsl>
    $<$<PLATFORM_ID:Android>:log>)
svtk_module_compile_features(SVTK::CommonCore
  PUBLIC
    cxx_std_11)

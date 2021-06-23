# Add the svtkDataArrayAPI tests. These are split up to separately test each
# array class, as testing all in a single translation unit led to heavy
# memory consumption and excessively long compile times.
set(data_array_tests)
function(add_data_array_test test_name array_header array_valuetype array_type)
  set(SVTK_TESTDATAARRAYAPI_HEADER "${array_header}")
  set(SVTK_TESTDATAARRAYAPI_VALUETYPE "${array_valuetype}")
  set(SVTK_TESTDATAARRAYAPI_ARRAYTYPE "${array_type}")
  set(SVTK_TESTDATAARRAYAPI_TESTNAME "TestDataArrayAPI_${test_name}")
  set(test_source "${SVTK_TESTDATAARRAYAPI_TESTNAME}.cxx")

  configure_file(TestDataArrayAPI.cxx.in "${CMAKE_CURRENT_BINARY_DIR}/${test_source}")

  #specify a custom test name as these files exist in the build directory
  list(APPEND data_array_tests "${SVTK_TESTDATAARRAYAPI_TESTNAME},${CMAKE_CURRENT_BINARY_DIR}/${test_source}" PARENT_SCOPE)
endfunction()

# Legacy -- test that the svtkDataArrayTemplate interface still works:
if(NOT SVTK_LEGACY_REMOVE)
  add_data_array_test(
    svtkDataArrayTemplate_LEGACY
    svtkDataArrayTemplate.h
    float
    svtkDataArrayTemplate<float>
  )
endif()

# Non-templated classes:
add_data_array_test(svtkCharArray svtkCharArray.h "char" svtkCharArray)
add_data_array_test(svtkDoubleArray svtkDoubleArray.h "double" svtkDoubleArray)
add_data_array_test(svtkFloatArray svtkFloatArray.h "float" svtkFloatArray)
add_data_array_test(svtkIntArray svtkIntArray.h "int" svtkIntArray)
add_data_array_test(svtkLongArray svtkLongArray.h "long" svtkLongArray)
add_data_array_test(svtkLongLongArray svtkLongLongArray.h "long long" svtkLongLongArray)
add_data_array_test(svtkShortArray svtkShortArray.h "short" svtkShortArray)
add_data_array_test(svtkSignedCharArray svtkSignedCharArray.h "signed char" svtkSignedCharArray)
add_data_array_test(svtkUnsignedCharArray svtkUnsignedCharArray.h "unsigned char" svtkUnsignedCharArray)
add_data_array_test(svtkUnsignedIntArray svtkUnsignedIntArray.h "unsigned int" svtkUnsignedIntArray)
add_data_array_test(svtkUnsignedLongArray svtkUnsignedLongArray.h "unsigned long" svtkUnsignedLongArray)
add_data_array_test(svtkUnsignedLongLongArray svtkUnsignedLongLongArray.h "unsigned long long" svtkUnsignedLongLongArray)
add_data_array_test(svtkUnsignedShortArray svtkUnsignedShortArray.h "unsigned short" svtkUnsignedShortArray)
add_data_array_test(svtkIdTypeArray svtkIdTypeArray.h "svtkIdType" svtkIdTypeArray)
set(value_types
  char
  double
  float
  int
  long
  "long long"
  short
  "signed char"
  "unsigned char"
  "unsigned int"
  "unsigned long"
  "unsigned long long"
  "unsigned short"
  svtkIdType
)

# Templated data array subclasses can be done in a loop:
foreach(value_type ${value_types})
  string(REPLACE " " "_" pretty_value_type "${value_type}")
  add_data_array_test(
    "AoS_${pretty_value_type}"
    svtkAOSDataArrayTemplate.h
    "${value_type}"
    "svtkAOSDataArrayTemplate<${value_type}>")
  add_data_array_test(
    "SoA_${pretty_value_type}"
    svtkSOADataArrayTemplate.h
    "${value_type}"
    "svtkSOADataArrayTemplate<${value_type}>")

  if(SVTK_BUILD_SCALED_SOA_ARRAYS)
    add_data_array_test(
      "ScaleAoS_${pretty_value_type}"
      svtkScaleAOSDataArrayTemplate.h
      "${value_type}"
      "svtkScaleAOSDataArrayTemplate<${value_type}>")
    add_data_array_test(
      "ScaleSoA_${pretty_value_type}"
      svtkScaledSOADataArrayTemplate.h
      "${value_type}"
      "svtkScaledSOADataArrayTemplate<${value_type}>")
  endif()
endforeach()

# Tell TestSystemInformation where to find the build trees.
set(TestSystemInformation_ARGS ${CMAKE_BINARY_DIR})

# Tell TestXMLFileOutputWindow where to write test file
set(TestXMLFileOutputWindow_ARGS ${CMAKE_BINARY_DIR}/Testing/Temporary/XMLFileOutputWindow.txt)

if (SVTK_BUILD_SCALED_SOA_ARRAYS)
  set(scale_soa_test TestScaledSOADataArrayTemplate.cxx)
endif ()


svtk_add_test_cxx(svtkCommonCoreCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  ExampleDataArrayRangeAPI.cxx
  ExampleDataArrayRangeDispatch.cxx
  UnitTestMath.cxx
  TestArrayAPI.cxx
  TestArrayAPIConvenience.cxx
  TestArrayAPIDense.cxx
  TestArrayAPISparse.cxx
  TestArrayBool.cxx
  TestArrayDispatchers.cxx
  TestAtomic.cxx
  TestScalarsToColors.cxx
  # TestArrayCasting.cxx # Uses Boost in its own separate test.
  TestArrayExtents.cxx
  TestArrayFreeFunctions.cxx
  TestArrayInterpolationDense.cxx
  TestArrayLookup.cxx
  TestArrayNullValues.cxx
  TestArraySize.cxx
  TestArrayUniqueValueDetection.cxx
  TestArrayUserTypes.cxx
  TestArrayVariants.cxx
  TestCollection.cxx
  TestConditionVariable.cxx
  # TestCxxFeatures.cxx # This is in its own exe too.
  TestDataArray.cxx
  TestDataArrayComponentNames.cxx
  TestDataArrayIterators.cxx
  TestDataArraySelection.cxx
  TestDataArrayTupleRange.cxx
  TestDataArrayValueRange.cxx
  TestGarbageCollector.cxx
  TestGenericDataArrayAPI.cxx
  TestInformationKeyLookup.cxx
  TestLogger.cxx
  TestLookupTable.cxx
  TestLookupTableThreaded.cxx
  TestMath.cxx
  TestMersenneTwister.cxx
  TestMinimalStandardRandomSequence.cxx
  TestNew.cxx
  TestNumberOfGenerationsFromBase.cxx
  TestObjectFactory.cxx
  TestObservers.cxx
  TestObserversPerformance.cxx
  TestOStreamWrapper.cxx
  TestSMP.cxx
  TestSmartPointer.cxx
  TestSortDataArray.cxx
  TestSparseArrayValidation.cxx
  TestSystemInformation.cxx
  TestTemplateMacro.cxx
  TestTimePointUtility.cxx
  TestUnicodeStringAPI.cxx
  TestUnicodeStringArrayAPI.cxx
  TestVariant.cxx
  TestVariantComparison.cxx
  TestWeakPointer.cxx
  TestXMLFileOutputWindow.cxx
  UnitTestInformationKeys.cxx
  otherArrays.cxx
  otherByteSwap.cxx
  # These two need svtkLookupTableWithEnabling/svtkLogLookupTable - move
  # them to RenderingCore or somewhere out there.
  #otherLookupTable.cxx
  #otherLookupTableWithEnabling.cxx
  otherStringArray.cxx

  ${data_array_tests}

  ${scale_soa_test}
  )

svtk_test_cxx_executable(svtkCommonCoreCxxTests tests
  svtkTestNewVar.cxx
  )

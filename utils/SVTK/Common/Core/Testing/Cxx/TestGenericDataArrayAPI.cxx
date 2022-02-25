/*==============================================================================

  Program:   Visualization Toolkit
  Module:    TestGenericDataArrayAPI.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

==============================================================================*/

#include "svtkGenericDataArray.h"

// Helpers:
#include "svtkAOSDataArrayTemplate.h"
#include "svtkArrayDispatch.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"
#include "svtkTypeTraits.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

// Concrete classes for testing:
#include "svtkAOSDataArrayTemplate.h"
#include "svtkCharArray.h"
#include "svtkDoubleArray.h"
#include "svtkFloatArray.h"
#include "svtkIdTypeArray.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkLongLongArray.h"
#include "svtkSOADataArrayTemplate.h"
#include "svtkShortArray.h"
#include "svtkSignedCharArray.h"
#ifdef SVTK_USE_SCALED_SOA_ARRAYS
#include "svtkScaledSOADataArrayTemplate.h"
#endif
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedIntArray.h"
#include "svtkUnsignedLongArray.h"
#include "svtkUnsignedLongLongArray.h"
#include "svtkUnsignedShortArray.h"

// About this test:
//
// This test runs a battery of unit tests that exercise the svtkGenericDataArray
// API on concrete implementations of their subclasses. It is designed to be
// easily extended to cover new array implementations and additional unit tests.
//
// This test has three main components:
// - Entry point: TestGenericDataArrayAPI(). Add new array classes here.
// - Unit test caller: ExerciseGenericDataArray(). Templated on value and array
//   types. Calls individual unit test functions to excerise the array methods.
//   Add new unit test calls here.
// - Unit test functions: Test_[methodSignature](). Templated on value type,
//   array type, and possibly other parameters to simplify implementations.
//   These should use the DataArrayAPI macros as needed

// Forward declare the test function:
namespace
{
template <typename ScalarT, typename ArrayT>
int ExerciseGenericDataArray();
} // end anon namespace

//------------------------------------------------------------------------------
//-------------Test Entry Point-------------------------------------------------
//------------------------------------------------------------------------------

int TestGenericDataArrayAPI(int, char*[])
{
  int errors = 0;

  // Add array classes here:
  // Defaults:
  errors += ExerciseGenericDataArray<char, svtkCharArray>();
  errors += ExerciseGenericDataArray<double, svtkDoubleArray>();
  errors += ExerciseGenericDataArray<float, svtkFloatArray>();
  errors += ExerciseGenericDataArray<int, svtkIntArray>();
  errors += ExerciseGenericDataArray<long, svtkLongArray>();
  errors += ExerciseGenericDataArray<long long, svtkLongLongArray>();
  errors += ExerciseGenericDataArray<short, svtkShortArray>();
  errors += ExerciseGenericDataArray<signed char, svtkSignedCharArray>();
  errors += ExerciseGenericDataArray<unsigned char, svtkUnsignedCharArray>();
  errors += ExerciseGenericDataArray<unsigned int, svtkUnsignedIntArray>();
  errors += ExerciseGenericDataArray<unsigned long, svtkUnsignedLongArray>();
  errors += ExerciseGenericDataArray<unsigned long long, svtkUnsignedLongLongArray>();
  errors += ExerciseGenericDataArray<unsigned short, svtkUnsignedShortArray>();
  errors += ExerciseGenericDataArray<svtkIdType, svtkIdTypeArray>();

  // Explicit AoS arrays:
  errors += ExerciseGenericDataArray<char, svtkAOSDataArrayTemplate<char> >();
  errors += ExerciseGenericDataArray<double, svtkAOSDataArrayTemplate<double> >();
  errors += ExerciseGenericDataArray<float, svtkAOSDataArrayTemplate<float> >();
  errors += ExerciseGenericDataArray<int, svtkAOSDataArrayTemplate<int> >();
  errors += ExerciseGenericDataArray<long, svtkAOSDataArrayTemplate<long> >();
  errors += ExerciseGenericDataArray<long long, svtkAOSDataArrayTemplate<long long> >();
  errors += ExerciseGenericDataArray<short, svtkAOSDataArrayTemplate<short> >();
  errors += ExerciseGenericDataArray<signed char, svtkAOSDataArrayTemplate<signed char> >();
  errors += ExerciseGenericDataArray<unsigned char, svtkAOSDataArrayTemplate<unsigned char> >();
  errors += ExerciseGenericDataArray<unsigned int, svtkAOSDataArrayTemplate<unsigned int> >();
  errors += ExerciseGenericDataArray<unsigned long, svtkAOSDataArrayTemplate<unsigned long> >();
  errors +=
    ExerciseGenericDataArray<unsigned long long, svtkAOSDataArrayTemplate<unsigned long long> >();
  errors += ExerciseGenericDataArray<unsigned short, svtkAOSDataArrayTemplate<unsigned short> >();
  errors += ExerciseGenericDataArray<svtkIdType, svtkAOSDataArrayTemplate<svtkIdType> >();

  // Explicit SoA arrays:
  errors += ExerciseGenericDataArray<char, svtkSOADataArrayTemplate<char> >();
  errors += ExerciseGenericDataArray<double, svtkSOADataArrayTemplate<double> >();
  errors += ExerciseGenericDataArray<float, svtkSOADataArrayTemplate<float> >();
  errors += ExerciseGenericDataArray<int, svtkSOADataArrayTemplate<int> >();
  errors += ExerciseGenericDataArray<long, svtkSOADataArrayTemplate<long> >();
  errors += ExerciseGenericDataArray<long long, svtkSOADataArrayTemplate<long long> >();
  errors += ExerciseGenericDataArray<short, svtkSOADataArrayTemplate<short> >();
  errors += ExerciseGenericDataArray<signed char, svtkSOADataArrayTemplate<signed char> >();
  errors += ExerciseGenericDataArray<unsigned char, svtkSOADataArrayTemplate<unsigned char> >();
  errors += ExerciseGenericDataArray<unsigned int, svtkSOADataArrayTemplate<unsigned int> >();
  errors += ExerciseGenericDataArray<unsigned long, svtkSOADataArrayTemplate<unsigned long> >();
  errors +=
    ExerciseGenericDataArray<unsigned long long, svtkSOADataArrayTemplate<unsigned long long> >();
  errors += ExerciseGenericDataArray<unsigned short, svtkSOADataArrayTemplate<unsigned short> >();
  errors += ExerciseGenericDataArray<svtkIdType, svtkSOADataArrayTemplate<svtkIdType> >();

  // Explicit scale SoA arrays:
#ifdef SVTK_USE_SCALED_SOA_ARRAYS
  errors += ExerciseGenericDataArray<char, svtkScaledSOADataArrayTemplate<char> >();
  errors += ExerciseGenericDataArray<double, svtkScaledSOADataArrayTemplate<double> >();
  errors += ExerciseGenericDataArray<float, svtkScaledSOADataArrayTemplate<float> >();
  errors += ExerciseGenericDataArray<int, svtkScaledSOADataArrayTemplate<int> >();
  errors += ExerciseGenericDataArray<long, svtkScaledSOADataArrayTemplate<long> >();
  errors += ExerciseGenericDataArray<long long, svtkScaledSOADataArrayTemplate<long long> >();
  errors += ExerciseGenericDataArray<short, svtkScaledSOADataArrayTemplate<short> >();
  errors += ExerciseGenericDataArray<signed char, svtkScaledSOADataArrayTemplate<signed char> >();
  errors +=
    ExerciseGenericDataArray<unsigned char, svtkScaledSOADataArrayTemplate<unsigned char> >();
  errors += ExerciseGenericDataArray<unsigned int, svtkScaledSOADataArrayTemplate<unsigned int> >();
  errors +=
    ExerciseGenericDataArray<unsigned long, svtkScaledSOADataArrayTemplate<unsigned long> >();
  errors += ExerciseGenericDataArray<unsigned long long,
    svtkScaledSOADataArrayTemplate<unsigned long long> >();
  errors +=
    ExerciseGenericDataArray<unsigned short, svtkScaledSOADataArrayTemplate<unsigned short> >();
  errors += ExerciseGenericDataArray<svtkIdType, svtkScaledSOADataArrayTemplate<svtkIdType> >();
#endif

  if (errors > 0)
  {
    std::cerr << "Test failed! Error count: " << errors << std::endl;
  }
  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

//------------------------------------------------------------------------------
//------------Unit Test Macros--------------------------------------------------
//------------------------------------------------------------------------------

#define DataArrayAPIInit(_signature)                                                               \
  int errors = 0;                                                                                  \
  std::string signature = _signature

#define DataArrayAPIUpdateSignature(_signature) signature = _signature

#define DataArrayAPIFinish() return errors

#define DataArrayAPICreateTestArray(name) svtkNew<ArrayT> name

#define DataArrayAPINonFatalError(x)                                                               \
  {                                                                                                \
    ArrayT* errorTempArray = ArrayT::New();                                                        \
    std::cerr << "Line " << __LINE__ << ": "                                                       \
              << "Failure in test of '" << signature << "' "                                       \
              << "for array type '" << errorTempArray->GetClassName() << "'"                       \
              << ":\n"                                                                             \
              << x << std::endl;                                                                   \
    errorTempArray->Delete();                                                                      \
    ++errors;                                                                                      \
  }

#define DataArrayAPIError(x) DataArrayAPINonFatalError(x) return errors;

namespace
{

//------------------------------------------------------------------------------
//------------------Unit Test Implementations-----------------------------------
//------------------------------------------------------------------------------

// ValueType GetValue(svtkIdType valueIdx) const
// No range checking/allocation.
template <typename ScalarT, typename ArrayT>
int Test_valT_GetValue_valueIdx_const()
{
  DataArrayAPIInit("ValueType GetValue(svtkIdType valueIdx) const");

  DataArrayAPICreateTestArray(array);
  svtkIdType comps = 9;
  svtkIdType tuples = 5;
  array->SetNumberOfComponents(comps);
  array->SetNumberOfTuples(tuples);

  // Initialize:
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    array->SetValue(i, static_cast<ScalarT>(i % 16));
  }

  // Verify:
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    ScalarT test = array->GetValue(i);
    ScalarT ref = static_cast<ScalarT>(i % 16);
    if (test != ref)
    {
      DataArrayAPIError("Data mismatch at value index '" << i << "'. Expected '" << ref
                                                         << "', got '" << test << "'.");
    }
  }

  DataArrayAPIFinish();
}

// void GetTypedTuple(svtkIdType tupleIdx, ValueType* tuple) const
template <typename ScalarT, typename ArrayT>
int Test_void_GetTypedTuple_tupleIdx_tuple()
{
  DataArrayAPIInit("void GetTypedTuple(svtkIdType tupleIdx, ValueType *tuple)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    source->SetValue(i, static_cast<ScalarT>(i % 17));
  }

  // Test the returned tuples:
  svtkIdType refValue = 0;
  std::vector<ScalarT> tuple(comps);
  for (svtkIdType tupleIdx = 0; tupleIdx < tuples; ++tupleIdx)
  {
    source->GetTypedTuple(tupleIdx, &tuple[0]);
    for (int compIdx = 0; compIdx < comps; ++compIdx)
    {
      if (tuple[compIdx] != static_cast<ScalarT>(refValue))
      {
        DataArrayAPIError("Data mismatch at tuple " << tupleIdx
                                                    << ", "
                                                       "component "
                                                    << compIdx << ": Expected '" << refValue
                                                    << "', got '" << tuple[compIdx] << "'.");
      }
      ++refValue;
      refValue %= 17;
    }
  }

  DataArrayAPIFinish();
}

// ValueType GetTypedComponent(svtkIdType tupleIdx, int comp) const
template <typename ScalarT, typename ArrayT>
int Test_valT_GetTypedComponent_tupleIdx_comp_const()
{
  DataArrayAPIInit("ValueType GetTypedComponent("
                   "svtkIdType tupleIdx, int comp) const");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    source->SetValue(i, static_cast<ScalarT>(i % 17));
  }

  // Test the returned tuples:
  svtkIdType refValue = 0;
  for (svtkIdType i = 0; i < tuples; ++i)
  {
    for (int j = 0; j < comps; ++j)
    {
      if (source->GetTypedComponent(i, j) != static_cast<ScalarT>(refValue))
      {
        DataArrayAPIError("Data mismatch at tuple " << i
                                                    << ", "
                                                       "component "
                                                    << j << ": Expected '" << refValue << "', got '"
                                                    << source->GetTypedComponent(i, j) << "'.");
      }
      ++refValue;
      refValue %= 17;
    }
  }

  DataArrayAPIFinish();
}

// void SetValue(svtkIdType valueIdx, ValueType value)
template <typename ScalarT, typename ArrayT>
int Test_void_SetValue_valueIdx_value()
{
  DataArrayAPIInit("void SetValue(svtkIdType valueIdx, ValueType value)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array using tested function:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    source->SetValue(i, static_cast<ScalarT>(((i + 1) * (i + 2)) % 17));
  }

  // Validate:
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    ScalarT ref = static_cast<ScalarT>(((i + 1) * (i + 2)) % 17);
    const typename ArrayT::ValueType test = source->GetValue(i);
    if (ref != test)
    {
      DataArrayAPIError(
        "Data mismatch at value " << i << ": Expected '" << ref << "', got '" << test << "'.");
    }
  }

  DataArrayAPIFinish();
}

// void SetTypedTuple(svtkIdType tupleIdx, const ValueType* tuple)
template <typename ScalarT, typename ArrayT>
int Test_void_SetTypedTuple_tupleIdx_tuple()
{
  DataArrayAPIInit("void SetTypedTuple(svtkIdType tupleIdx, "
                   "const ValueType* tuple)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 5;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    std::vector<ScalarT> tuple;
    tuple.reserve(comps);
    for (int c = 0; c < comps; ++c)
    {
      tuple.push_back(static_cast<ScalarT>(((t * comps) + c) % 17));
    }
    source->SetTypedTuple(t, &tuple[0]);
  }

  // Verify:
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    for (int c = 0; c < comps; ++c)
    {
      ScalarT ref = static_cast<ScalarT>(((t * comps) + c) % 17);
      ScalarT test = source->GetTypedComponent(t, c);
      if (ref != test)
      {
        DataArrayAPIError("Data mismatch at tuple " << t << " component " << c << ": Expected "
                                                    << ref << ", got " << test << ".");
      }
    }
  }

  DataArrayAPIFinish();
}

// void SetTypedComponent(svtkIdType tupleIdx, int comp, ValueType value)
template <typename ScalarT, typename ArrayT>
int Test_void_SetTypedComponent_tupleIdx_comp_value()
{
  DataArrayAPIInit("void SetTypedComponent(svtkIdType tupleIdx, int comp, "
                   "ValueType value)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array using tested function:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);
  for (svtkIdType i = 0; i < tuples; ++i)
  {
    for (int j = 0; j < comps; ++j)
    {
      source->SetTypedComponent(i, j, static_cast<ScalarT>(((i + 1) * (j + 1)) % 17));
    }
  }

  // Test the returned tuples:
  std::vector<ScalarT> tuple(comps);
  for (svtkIdType i = 0; i < tuples; ++i)
  {
    source->GetTypedTuple(i, &tuple[0]);
    for (int j = 0; j < comps; ++j)
    {
      ScalarT test = tuple[j];
      ScalarT ref = static_cast<ScalarT>((i + 1) * (j + 1) % 17);
      if (ref != test)
      {
        DataArrayAPIError("Data mismatch at tuple " << i << ", component " << j << ": Expected '"
                                                    << ref << "', got '" << test << "'.");
      }
    }
  }

  DataArrayAPIFinish();
}

// svtkIdType LookupTypedValue(ValueType value)
// void LookupTypedValue(ValueType value, svtkIdList* ids)
template <typename ScalarT, typename ArrayT>
int Test_LookupTypedValue_allSigs()
{
  DataArrayAPIInit("LookupTypedValue");

  DataArrayAPICreateTestArray(array);

  // Map Value --> ValueIdxs. We'll use this to validate the lookup results.
  typedef std::map<ScalarT, svtkIdList*> RefMap;
  typedef typename RefMap::iterator RefMapIterator;
  RefMap refMap;
  // These are the values we'll be looking for.
  for (ScalarT val = 0; val < 17; ++val)
  {
    refMap.insert(std::make_pair(val, svtkIdList::New()));
  }

  // Initialize source array:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  array->SetNumberOfComponents(comps);
  array->SetNumberOfTuples(tuples);
  for (svtkIdType valIdx = 0; valIdx < comps * tuples; ++valIdx)
  {
    ScalarT val = static_cast<ScalarT>(valIdx % 17);
    array->SetValue(valIdx, val);
    // Update our reference map:
    RefMapIterator it = refMap.find(val);
    assert("Value exists in reference map." && it != refMap.end());
    it->second->InsertNextId(valIdx);
  }

  // Test the lookup functions.
  svtkNew<svtkIdList> testIdList;
  for (RefMapIterator it = refMap.begin(), itEnd = refMap.end(); it != itEnd; ++it)
  {
    const ScalarT& val = it->first;
    svtkIdList* refIdList = it->second; // Presorted due to insertion order
    svtkIdType* refIdBegin = refIdList->GetPointer(0);
    svtkIdType* refIdEnd = refIdList->GetPointer(refIdList->GetNumberOfIds());

    // Docs are unclear about this. Does it return the first value, or just any?
    // We'll assume any since it's unspecified.
    DataArrayAPIUpdateSignature("svtkIdType LookupTypedValue(ValueType value)");
    svtkIdType testId = array->LookupTypedValue(val);
    if (!std::binary_search(refIdBegin, refIdEnd, testId))
    {
      // NonFatal + break so we can clean up.
      DataArrayAPINonFatalError("Looking up value '" << val << "' returned valueIdx '" << testId
                                                     << "', which maps to value '"
                                                     << array->GetValue(testId) << "'.");
      break;
    }

    // Now for the list overload:
    DataArrayAPIUpdateSignature("void LookupTypedValue(ValueType value, svtkIdList* ids)");
    array->LookupTypedValue(val, testIdList);
    if (testIdList->GetNumberOfIds() != refIdList->GetNumberOfIds())
    {
      // NonFatal + break so we can clean up.
      DataArrayAPINonFatalError("Looking up value '"
        << val << "' returned " << testIdList->GetNumberOfIds() << " ids, but "
        << refIdList->GetNumberOfIds() << "were expected.");
      break;
    }
    svtkIdType* testIdBegin = testIdList->GetPointer(0);
    svtkIdType* testIdEnd = testIdList->GetPointer(refIdList->GetNumberOfIds());
    // Ensure the test ids are sorted
    std::sort(testIdBegin, testIdEnd);
    if (!std::equal(testIdBegin, testIdEnd, refIdBegin))
    {
      // NonFatal + break so we can clean up.
      DataArrayAPINonFatalError("Looking up all value indices for value '"
        << val << "' did not return the expected result.");
      break;
    }
  }

  // Cleanup:
  for (RefMapIterator it = refMap.begin(), itEnd = refMap.end(); it != itEnd; ++it)
  {
    it->second->Delete();
    it->second = nullptr;
  }

  DataArrayAPIFinish();
}

// svtkIdType InsertNextValue(ValueType v)
template <typename ScalarT, typename ArrayT>
int Test_svtkIdType_InsertNextValue_v()
{
  DataArrayAPIInit("svtkIdType InsertNextValue(ValueType v)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array using tested function:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    svtkIdType insertLoc = source->InsertNextValue(static_cast<ScalarT>(i % 17));
    if (insertLoc != i)
    {
      DataArrayAPIError(
        "Returned location incorrect. Expected '" << i << "', got '" << insertLoc << "'.");
    }
    if (source->GetSize() < i + 1)
    {
      DataArrayAPIError(
        "Size should be at least " << i + 1 << " values, but is only " << source->GetSize() << ".");
    }
    if (source->GetMaxId() != i)
    {
      DataArrayAPIError(
        "MaxId should be " << i << ", but is " << source->GetMaxId() << " instead.");
    }
  }

  // Validate:
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    ScalarT ref = static_cast<ScalarT>(i % 17);
    const typename ArrayT::ValueType test = source->GetValue(i);
    if (ref != test)
    {
      DataArrayAPIError(
        "Data mismatch at value " << i << ": Expected '" << ref << "', got '" << test << "'.");
    }
  }

  DataArrayAPIFinish();
}

// void InsertValue(svtkIdType idx, ValueType v)
template <typename ScalarT, typename ArrayT>
int Test_void_InsertValue_idx_v()
{
  DataArrayAPIInit("void InsertValue(svtkIdType idx, ValueType v)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array using tested function:
  svtkIdType comps = 9;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    source->InsertValue(i, static_cast<ScalarT>(i % 17));

    if (source->GetSize() < i + 1)
    {
      DataArrayAPIError(
        "Size should be at least " << i + 1 << " values, but is only " << source->GetSize() << ".");
    }
    if (source->GetMaxId() != i)
    {
      DataArrayAPIError(
        "MaxId should be " << i << ", but is " << source->GetMaxId() << " instead.");
    }
  }

  // Validate:
  for (svtkIdType i = 0; i < comps * tuples; ++i)
  {
    ScalarT ref = static_cast<ScalarT>(i % 17);
    const typename ArrayT::ValueType test = source->GetValue(i);
    if (ref != test)
    {
      DataArrayAPIError(
        "Data mismatch at value " << i << ": Expected '" << ref << "', got '" << test << "'.");
    }
  }

  DataArrayAPIFinish();
}

// void InsertTypedTuple(svtkIdType idx, const ValueType *t)
template <typename ScalarT, typename ArrayT>
int Test_void_InsertTypedTuple_idx_t()
{
  DataArrayAPIInit("void InsertTypedTuple(svtkIdType idx, const ValueType *t)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 5;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    std::vector<ScalarT> tuple;
    tuple.reserve(comps);
    for (int c = 0; c < comps; ++c)
    {
      tuple.push_back(static_cast<ScalarT>(((t * comps) + c) % 17));
    }
    source->InsertTypedTuple(t, &tuple[0]);
    if (source->GetSize() < ((t + 1) * comps))
    {
      DataArrayAPIError("Size should be at least " << ((t + 1) * comps) << " values, but is only "
                                                   << source->GetSize() << ".");
    }
    if (source->GetMaxId() != ((t + 1) * comps) - 1)
    {
      DataArrayAPIError("MaxId should be " << ((t + 1) * comps) - 1 << ", but is "
                                           << source->GetMaxId() << " instead.");
    }
  }

  // Verify:
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    for (int c = 0; c < comps; ++c)
    {
      if (source->GetTypedComponent(t, c) != static_cast<ScalarT>(((t * comps) + c) % 17))
      {
        DataArrayAPIError("Data mismatch at tuple " << t << " component " << c << ": Expected "
                                                    << static_cast<ScalarT>(((t * comps) + c) % 17)
                                                    << ", got " << source->GetTypedComponent(t, c)
                                                    << ".");
      }
    }
  }

  DataArrayAPIFinish();
}

// svtkIdType InsertNextTypedTuple(const ValueType *t)
template <typename ScalarT, typename ArrayT>
int Test_svtkIdType_InsertNextTypedTuple_t()
{
  DataArrayAPIInit("svtkIdType InsertNextTypedTuple(const ValueType *t)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 5;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    std::vector<ScalarT> tuple;
    tuple.reserve(comps);
    for (int c = 0; c < comps; ++c)
    {
      tuple.push_back(static_cast<ScalarT>(((t * comps) + c) % 17));
    }
    svtkIdType insertLoc = source->InsertNextTypedTuple(&tuple[0]);
    if (insertLoc != t)
    {
      DataArrayAPIError(
        "Returned location incorrect. Expected '" << t << "', got '" << insertLoc << "'.");
    }
    if (source->GetSize() < ((t + 1) * comps))
    {
      DataArrayAPIError("Size should be at least " << ((t + 1) * comps) << " values, but is only "
                                                   << source->GetSize() << ".");
    }
    if (source->GetMaxId() != ((t + 1) * comps) - 1)
    {
      DataArrayAPIError("MaxId should be " << ((t + 1) * comps) - 1 << ", but is "
                                           << source->GetMaxId() << " instead.");
    }
  }

  // Verify:
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    for (int c = 0; c < comps; ++c)
    {
      if (source->GetTypedComponent(t, c) != static_cast<ScalarT>(((t * comps) + c) % 17))
      {
        DataArrayAPIError("Data mismatch at tuple " << t << " component " << c << ": Expected "
                                                    << static_cast<ScalarT>(((t * comps) + c) % 17)
                                                    << ", got " << source->GetTypedComponent(t, c)
                                                    << ".");
      }
    }
  }

  DataArrayAPIFinish();
}

// svtkIdType GetNumberOfValues()
template <typename ScalarT, typename ArrayT>
int Test_svtkIdType_GetNumberOfValues()
{
  DataArrayAPIInit("svtkIdType InsertNextTypedTuple(const ValueType *t)");

  DataArrayAPICreateTestArray(source);

  // Initialize source array:
  svtkIdType comps = 5;
  svtkIdType tuples = 10;
  source->SetNumberOfComponents(comps);
  source->SetNumberOfTuples(tuples);

  if (source->GetNumberOfValues() != comps * tuples)
  {
    DataArrayAPIError("Returned number of values: " << source->GetNumberOfValues() << ", expected "
                                                    << (comps * tuples) << ".");
  }

  DataArrayAPIFinish();
}

// ValueType *GetValueRange()
// void GetValueRange(ValueType range[2])
// ValueType *GetValueRange(int comp)
// void GetValueRange(ValueType range[2], int comp)
template <typename ScalarT, typename ArrayT>
int Test_GetValueRange_all_overloads()
{
  DataArrayAPIInit("GetValueRange");

  DataArrayAPICreateTestArray(array);

  // Initialize arrays:
  int comps = 6;
  svtkIdType tuples = 9;
  array->SetNumberOfComponents(comps);
  array->SetNumberOfTuples(tuples);
  for (svtkIdType t = 0; t < tuples; ++t)
  {
    for (int c = 0; c < comps; ++c)
    {
      array->SetComponent(t, c, (t + 1) * (c + 1));
    }
  }

  // Create a copy of the test array, but set some values to the min/max of
  // the data type to ensure that the full range is supported:
  DataArrayAPICreateTestArray(arrayMinMax);
  arrayMinMax->DeepCopy(array);
  assert(comps < tuples && "The logic below assumes more tuples than comps");
  assert(comps % 2 == 0 && "The logic below assumes an even number of comps");
  for (int c = 0; c < comps; ++c)
  {
    arrayMinMax->SetTypedComponent(static_cast<svtkIdType>(c), c, svtkTypeTraits<ScalarT>::Min());
    arrayMinMax->SetTypedComponent(
      static_cast<svtkIdType>(c), comps - c - 1, svtkTypeTraits<ScalarT>::Max());
  }

  // Just the range of the first component:
  DataArrayAPIUpdateSignature("ValueType* GetValueRange()");
  ScalarT* rangePtr = array->GetValueRange();
  ScalarT expectedRange[2] = { static_cast<ScalarT>(1), static_cast<ScalarT>(tuples) };
  if (rangePtr[0] != expectedRange[0] || rangePtr[1] != expectedRange[1])
  {
    DataArrayAPINonFatalError("First component range expected to be: ["
      << expectedRange[0] << ", " << expectedRange[1] << "], got [" << rangePtr[0] << ", "
      << rangePtr[1] << "].");
  }

  rangePtr = arrayMinMax->GetValueRange();
  if (rangePtr[0] != svtkTypeTraits<ScalarT>::Min() || rangePtr[1] != svtkTypeTraits<ScalarT>::Max())
  {
    DataArrayAPINonFatalError("First component range expected to be: ["
      << svtkTypeTraits<ScalarT>::Min() << ", " << svtkTypeTraits<ScalarT>::Max() << "], got ["
      << rangePtr[0] << ", " << rangePtr[1] << "].");
  }

  DataArrayAPIUpdateSignature("void GetValueRange(ValueType range[2])");
  ScalarT rangeArray[2];
  array->GetValueRange(rangeArray);
  if (rangeArray[0] != expectedRange[0] || rangeArray[1] != expectedRange[1])
  {
    DataArrayAPINonFatalError("First component range expected to be: ["
      << expectedRange[0] << ", " << expectedRange[1] << "], got [" << rangeArray[0] << ", "
      << rangeArray[1] << "].");
  }

  arrayMinMax->GetValueRange(rangeArray);
  if (rangeArray[0] != svtkTypeTraits<ScalarT>::Min() ||
    rangeArray[1] != svtkTypeTraits<ScalarT>::Max())
  {
    DataArrayAPINonFatalError("First component range expected to be: ["
      << svtkTypeTraits<ScalarT>::Min() << ", " << svtkTypeTraits<ScalarT>::Max() << "], got ["
      << rangePtr[0] << ", " << rangePtr[1] << "].");
  }

  DataArrayAPIUpdateSignature("ValueType* GetValueRange(int comp)");
  for (int c = 0; c < comps; ++c)
  {
    expectedRange[0] = static_cast<ScalarT>(c + 1);
    expectedRange[1] = static_cast<ScalarT>(tuples * (c + 1));
    rangePtr = array->GetValueRange(c);
    if (rangePtr[0] != expectedRange[0] || rangePtr[1] != expectedRange[1])
    {
      DataArrayAPINonFatalError("Component " << c << " range expected to be: [" << expectedRange[0]
                                             << ", " << expectedRange[1] << "], got ["
                                             << rangePtr[0] << ", " << rangePtr[1] << "].");
    }

    rangePtr = arrayMinMax->GetValueRange(c);
    if (rangePtr[0] != svtkTypeTraits<ScalarT>::Min() ||
      rangePtr[1] != svtkTypeTraits<ScalarT>::Max())
    {
      DataArrayAPINonFatalError("Component "
        << c << " range expected to be: [" << svtkTypeTraits<ScalarT>::Min() << ", "
        << svtkTypeTraits<ScalarT>::Max() << "], got [" << rangePtr[0] << ", " << rangePtr[1]
        << "].");
    }
  }

  DataArrayAPIUpdateSignature("void GetValueRange(ValueType range[2], int comp)");
  for (int c = 0; c < comps; ++c)
  {
    expectedRange[0] = static_cast<ScalarT>(c + 1);
    expectedRange[1] = static_cast<ScalarT>(tuples * (c + 1));
    array->GetValueRange(rangeArray, c);
    if (rangeArray[0] != expectedRange[0] || rangeArray[1] != expectedRange[1])
    {
      DataArrayAPINonFatalError("Component " << c << " range expected to be: [" << expectedRange[0]
                                             << ", " << expectedRange[1] << "], got ["
                                             << rangeArray[0] << ", " << rangeArray[1] << "].");
    }

    arrayMinMax->GetValueRange(rangeArray, c);
    if (rangeArray[0] != svtkTypeTraits<ScalarT>::Min() ||
      rangeArray[1] != svtkTypeTraits<ScalarT>::Max())
    {
      DataArrayAPINonFatalError("Component "
        << c << " range expected to be: [" << svtkTypeTraits<ScalarT>::Min() << ", "
        << svtkTypeTraits<ScalarT>::Max() << "], got [" << rangePtr[0] << ", " << rangePtr[1]
        << "].");
    }
  }

  DataArrayAPIFinish();
}

//------------------------------------------------------------------------------
//-----------Unit Test Function Caller------------------------------------------
//------------------------------------------------------------------------------

template <typename ScalarT, typename ArrayT>
int ExerciseGenericDataArray()
{
  int errors = 0;

  errors += Test_valT_GetValue_valueIdx_const<ScalarT, ArrayT>();
  errors += Test_void_GetTypedTuple_tupleIdx_tuple<ScalarT, ArrayT>();
  errors += Test_valT_GetTypedComponent_tupleIdx_comp_const<ScalarT, ArrayT>();
  errors += Test_void_SetValue_valueIdx_value<ScalarT, ArrayT>();
  errors += Test_void_SetTypedTuple_tupleIdx_tuple<ScalarT, ArrayT>();
  errors += Test_void_SetTypedComponent_tupleIdx_comp_value<ScalarT, ArrayT>();
  errors += Test_LookupTypedValue_allSigs<ScalarT, ArrayT>();
  errors += Test_svtkIdType_InsertNextValue_v<ScalarT, ArrayT>();
  errors += Test_void_InsertValue_idx_v<ScalarT, ArrayT>();
  errors += Test_void_InsertTypedTuple_idx_t<ScalarT, ArrayT>();
  errors += Test_svtkIdType_InsertNextTypedTuple_t<ScalarT, ArrayT>();
  errors += Test_svtkIdType_GetNumberOfValues<ScalarT, ArrayT>();
  errors += Test_GetValueRange_all_overloads<ScalarT, ArrayT>();

  return errors;
} // end ExerciseDataArray

} // end anon namespace

#undef DataArrayAPIInit
#undef DataArrayAPIUpdateSignature
#undef DataArrayAPIFinish
#undef DataArrayAPICreateTestArray
#undef DataArrayAPINonFatalError
#undef DataArrayAPIError

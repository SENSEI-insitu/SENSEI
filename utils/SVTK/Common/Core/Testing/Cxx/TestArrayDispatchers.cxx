/*==============================================================================

  Program:   Visualization Toolkit
  Module:    TestArrayDispatchers.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

==============================================================================*/

// We define our own set of arrays for the dispatch list. This allows the test
// to run regardless of the compiled dispatch configuration. Note that this is
// only possible because we do not use dispatches that are compiled into other
// translation units, but only explicit dispatches that we generate here.
#define svtkArrayDispatchArrayList_h // Skip loading the actual header

#include "svtkAOSDataArrayTemplate.h"
#include "svtkSOADataArrayTemplate.h"
#include "svtkTypeList.h"

#include <type_traits> // for std::is_[lr]value_reference
#include <utility>     // for std::move

namespace svtkArrayDispatch
{
typedef svtkTypeList::Unique<                //
  svtkTypeList::Create<                      //
    svtkAOSDataArrayTemplate<double>,        //
    svtkAOSDataArrayTemplate<float>,         //
    svtkAOSDataArrayTemplate<int>,           //
    svtkAOSDataArrayTemplate<unsigned char>, //
    svtkAOSDataArrayTemplate<svtkIdType>,     //
    svtkSOADataArrayTemplate<double>,        //
    svtkSOADataArrayTemplate<float>,         //
    svtkSOADataArrayTemplate<int>,           //
    svtkSOADataArrayTemplate<unsigned char>, //
    svtkSOADataArrayTemplate<svtkIdType>      //
    > >::Result Arrays;
} // end namespace svtkArrayDispatch

#include "svtkArrayDispatch.h"
#include "svtkNew.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace
{

//==============================================================================
// Our functor for testing.
struct TestWorker
{
  TestWorker()
    : Array1(nullptr)
    , Array2(nullptr)
    , Array3(nullptr)
  {
  }

  void Reset()
  {
    this->Array1 = nullptr;
    this->Array2 = nullptr;
    this->Array3 = nullptr;
  }

  template <typename Array1T>
  void operator()(Array1T* array1)
  {
    this->Array1 = array1;
  }

  template <typename Array1T, typename Array2T>
  void operator()(Array1T* array1, Array2T* array2)
  {
    this->Array1 = array1;
    this->Array2 = array2;
  }

  template <typename Array1T, typename Array2T, typename Array3T>
  void operator()(Array1T* array1, Array2T* array2, Array3T* array3)
  {
    this->Array1 = array1;
    this->Array2 = array2;
    this->Array3 = array3;
  }

  svtkDataArray* Array1;
  svtkDataArray* Array2;
  svtkDataArray* Array3;
};

//==============================================================================
// Functor for testing parameter forwarding.
struct ForwardedParams
{
  bool Success{ false };

  template <typename ArrayT, typename T1, typename T2>
  void operator()(ArrayT*, T1&& t1, T2&& t2)
  {
    this->Success = std::is_lvalue_reference<T1&&>::value &&
      std::is_rvalue_reference<T2&&>::value && t1 == 42 && t2 == 20;
  }

  template <typename ArrayT1, typename ArrayT2, typename T1, typename T2>
  void operator()(ArrayT1*, ArrayT2*, T1&& t1, T2&& t2)
  {
    this->Success = std::is_lvalue_reference<T1&&>::value &&
      std::is_rvalue_reference<T2&&>::value && t1 == 42 && t2 == 20;
  }

  template <typename ArrayT1, typename ArrayT2, typename ArrayT3, typename T1, typename T2>
  void operator()(ArrayT1*, ArrayT2*, ArrayT3*, T1&& t1, T2&& t2)
  {
    this->Success = std::is_lvalue_reference<T1&&>::value &&
      std::is_rvalue_reference<T2&&>::value && t1 == 42 && t2 == 20;
  }

  void Reset() { this->Success = false; }
};

//==============================================================================
// Functor to test that rvalue functors work:
bool ForwardedFunctorCalled = false; // global for validating calls
struct ForwardedFunctor
{
  void operator()(...) const
  {
    assert(!ForwardedFunctorCalled);
    ForwardedFunctorCalled = true;
  }
};

//==============================================================================
// Container for testing arrays.
struct Arrays
{
  Arrays();
  ~Arrays();

  static svtkAOSDataArrayTemplate<double>* aosDouble;
  static svtkAOSDataArrayTemplate<float>* aosFloat;
  static svtkAOSDataArrayTemplate<int>* aosInt;
  static svtkAOSDataArrayTemplate<unsigned char>* aosUnsignedChar;
  static svtkAOSDataArrayTemplate<svtkIdType>* aosIdType;

  static svtkSOADataArrayTemplate<double>* soaDouble;
  static svtkSOADataArrayTemplate<float>* soaFloat;
  static svtkSOADataArrayTemplate<int>* soaInt;
  static svtkSOADataArrayTemplate<unsigned char>* soaUnsignedChar;
  static svtkSOADataArrayTemplate<svtkIdType>* soaIdType;

  static std::vector<svtkDataArray*> aosArrays;
  static std::vector<svtkDataArray*> soaArrays;
  static std::vector<svtkDataArray*> allArrays;
};

svtkAOSDataArrayTemplate<double>* Arrays::aosDouble;
svtkAOSDataArrayTemplate<float>* Arrays::aosFloat;
svtkAOSDataArrayTemplate<int>* Arrays::aosInt;
svtkAOSDataArrayTemplate<unsigned char>* Arrays::aosUnsignedChar;
svtkAOSDataArrayTemplate<svtkIdType>* Arrays::aosIdType;
svtkSOADataArrayTemplate<double>* Arrays::soaDouble;
svtkSOADataArrayTemplate<float>* Arrays::soaFloat;
svtkSOADataArrayTemplate<int>* Arrays::soaInt;
svtkSOADataArrayTemplate<unsigned char>* Arrays::soaUnsignedChar;
svtkSOADataArrayTemplate<svtkIdType>* Arrays::soaIdType;

std::vector<svtkDataArray*> Arrays::aosArrays;
std::vector<svtkDataArray*> Arrays::soaArrays;
std::vector<svtkDataArray*> Arrays::allArrays;

//==============================================================================
// Miscellaneous Debris
typedef std::vector<svtkDataArray*>::iterator ArrayIter;

typedef svtkTypeList::Create<              //
  svtkAOSDataArrayTemplate<double>,        //
  svtkAOSDataArrayTemplate<float>,         //
  svtkAOSDataArrayTemplate<int>,           //
  svtkAOSDataArrayTemplate<unsigned char>, //
  svtkAOSDataArrayTemplate<svtkIdType>      //
  >
  AoSArrayList;
typedef svtkTypeList::Create<              //
  svtkSOADataArrayTemplate<double>,        //
  svtkSOADataArrayTemplate<float>,         //
  svtkSOADataArrayTemplate<int>,           //
  svtkSOADataArrayTemplate<unsigned char>, //
  svtkSOADataArrayTemplate<svtkIdType>      //
  >
  SoAArrayList;

typedef svtkTypeList::Append<AoSArrayList, SoAArrayList>::Result AllArrayList;

//------------------------------------------------------------------------------
// Return true if the SVTK type tag is an integral type.
inline bool isIntegral(int svtkType)
{
  switch (svtkType)
  {
    case SVTK_CHAR:
    case SVTK_SIGNED_CHAR:
    case SVTK_UNSIGNED_CHAR:
    case SVTK_SHORT:
    case SVTK_UNSIGNED_SHORT:
    case SVTK_INT:
    case SVTK_UNSIGNED_INT:
    case SVTK_LONG:
    case SVTK_UNSIGNED_LONG:
    case SVTK_ID_TYPE:
    case SVTK_LONG_LONG:
    case SVTK_UNSIGNED_LONG_LONG:
#if !defined(SVTK_LEGACY_REMOVE)
    case SVTK___INT64:
    case SVTK_UNSIGNED___INT64:
#endif
      return true;
  }
  return false;
}

//------------------------------------------------------------------------------
// Return true if the SVTK type tag is a real (e.g. floating-point) type.
inline bool isReal(int svtkType)
{
  switch (svtkType)
  {
    case SVTK_FLOAT:
    case SVTK_DOUBLE:
      return true;
  }
  return false;
}

//------------------------------------------------------------------------------
// Check condition during test.
#define testAssert(expr, errorMessage)                                                             \
  if (!(expr))                                                                                     \
  {                                                                                                \
    ++errors;                                                                                      \
    svtkGenericWarningMacro(<< "Assertion failed: " #expr << "\n" << errorMessage);                 \
  }                                                                                                \
  []() {}() /* Swallow semi-colon */

//------------------------------------------------------------------------------
int TestDispatch()
{
  int errors = 0;

  using Dispatcher = svtkArrayDispatch::Dispatch;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array : Arrays::allArrays)
  {
    testAssert(Dispatcher::Execute(array, worker), "Dispatch failed.");
    testAssert(worker.Array1 == array, "Array 1 does not match input.");
    worker.Reset();

    int lval{ 42 };
    int rval{ 20 };
    testAssert(Dispatcher::Execute(array, paramTester, lval, std::move(rval)),
      "Parameter forwarding dispatch failed.");
    testAssert(paramTester.Success, "Parameter forwarding failed.");
    paramTester.Reset();

    testAssert(
      Dispatcher::Execute(array, ForwardedFunctor{}), "Functor forwarding dispatch failed.");
    testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
    ForwardedFunctorCalled = false;
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatchByArray()
{
  int errors = 0;

  using Dispatcher = svtkArrayDispatch::DispatchByArray<AoSArrayList>;
  TestWorker worker;
  ForwardedParams paramTester;

  // AoS arrays: All should pass:
  for (svtkDataArray* array : Arrays::aosArrays)
  {
    testAssert(Dispatcher::Execute(array, worker), "Dispatch failed.");
    testAssert(worker.Array1 == array, "Array 1 does not match input.");
    worker.Reset();

    int lval{ 42 };
    int rval{ 20 };
    testAssert(Dispatcher::Execute(array, paramTester, lval, std::move(rval)),
      "Parameter forwarding dispatch failed.");
    testAssert(paramTester.Success, "Parameter forwarding failed.");
    paramTester.Reset();

    testAssert(
      Dispatcher::Execute(array, ForwardedFunctor{}), "Functor forwarding dispatch failed.");
    testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
    ForwardedFunctorCalled = false;
  }

  // AoS arrays: All should fail:
  for (ArrayIter it = Arrays::soaArrays.begin(), itEnd = Arrays::soaArrays.end(); it != itEnd; ++it)
  {
    svtkDataArray* array = *it;
    testAssert(!Dispatcher::Execute(array, worker), "Dispatch should have failed.");
    testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
    worker.Reset();
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatchByValueType()
{
  int errors = 0;

  // Create dispatcher that only generates code paths for arrays with reals.
  using Dispatcher = svtkArrayDispatch::DispatchByValueType<svtkArrayDispatch::Reals>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array : Arrays::allArrays)
  {
    bool isValid = isReal(array->GetDataType());

    if (isValid)
    {
      testAssert(Dispatcher::Execute(array, worker), "Dispatch failed.");
      testAssert(worker.Array1 == array, "Array 1 does not match input.");
      worker.Reset();

      int lval{ 42 };
      int rval{ 20 };
      testAssert(Dispatcher::Execute(array, paramTester, lval, std::move(rval)),
        "Parameter forwarding dispatch failed.");
      testAssert(paramTester.Success, "Parameter forwarding failed.");
      paramTester.Reset();

      testAssert(
        Dispatcher::Execute(array, ForwardedFunctor{}), "Functor forwarding dispatch failed.");
      testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
      ForwardedFunctorCalled = false;
    }
    else
    {
      testAssert(!Dispatcher::Execute(array, worker), "Dispatch should have failed.");
      testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
      worker.Reset();
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch2ByArray()
{
  int errors = 0;

  // Restrictions:
  // Array1: SoA
  // Array2: AoS
  using Dispatcher = svtkArrayDispatch::Dispatch2ByArray<SoAArrayList, AoSArrayList>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = array1->GetArrayType() == svtkAbstractArray::SoADataArrayTemplate;

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = array2->GetArrayType() == svtkAbstractArray::AoSDataArrayTemplate;

      if (array1Valid && array2Valid)
      {
        testAssert(Dispatcher::Execute(array1, array2, worker), "Dispatch failed.");
        testAssert(worker.Array1 == array1, "Array 1 does not match input.");
        testAssert(worker.Array2 == array2, "Array 2 does not match input.");
        worker.Reset();

        int lval{ 42 };
        int rval{ 20 };
        testAssert(Dispatcher::Execute(array1, array2, paramTester, lval, std::move(rval)),
          "Parameter forwarding dispatch failed.");
        testAssert(paramTester.Success, "Parameter forwarding failed.");
        paramTester.Reset();

        testAssert(Dispatcher::Execute(array1, array2, ForwardedFunctor{}),
          "Functor forwarding dispatch failed.");
        testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
        ForwardedFunctorCalled = false;
      }
      else
      {
        testAssert(!Dispatcher::Execute(array1, array2, worker), "Dispatch should have failed.");
        testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
        testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
        worker.Reset();
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch2ByValueType()
{
  int errors = 0;

  // Restrictions:
  // Array1: Integers
  // Array2: Reals
  using Dispatcher =
    svtkArrayDispatch::Dispatch2ByValueType<svtkArrayDispatch::Integrals, svtkArrayDispatch::Reals>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = isIntegral(array1->GetDataType());

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = isReal(array2->GetDataType());

      if (array1Valid && array2Valid)
      {
        testAssert(Dispatcher::Execute(array1, array2, worker), "Dispatch failed.");
        testAssert(worker.Array1 == array1, "Array 1 does not match input.");
        testAssert(worker.Array2 == array2, "Array 2 does not match input.");
        worker.Reset();

        int lval{ 42 };
        int rval{ 20 };
        testAssert(Dispatcher::Execute(array1, array2, paramTester, lval, std::move(rval)),
          "Parameter forwarding dispatch failed.");
        testAssert(paramTester.Success, "Parameter forwarding failed.");
        paramTester.Reset();

        testAssert(Dispatcher::Execute(array1, array2, ForwardedFunctor{}),
          "Functor forwarding dispatch failed.");
        testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
        ForwardedFunctorCalled = false;
      }
      else
      {
        testAssert(!Dispatcher::Execute(array1, array2, worker), "Dispatch should have failed.");
        testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
        testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
        worker.Reset();
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch2ByArrayWithSameValueType()
{
  int errors = 0;

  // Restrictions:
  // - Types must match
  using Dispatcher =
    svtkArrayDispatch::Dispatch2ByArrayWithSameValueType<AoSArrayList, SoAArrayList>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = array1->GetArrayType() == svtkAbstractArray::AoSDataArrayTemplate;

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = array2->GetArrayType() == svtkAbstractArray::SoADataArrayTemplate &&
        svtkDataTypesCompare(array1->GetDataType(), array2->GetDataType());

      if (array1Valid && array2Valid)
      {
        testAssert(Dispatcher::Execute(array1, array2, worker), "Dispatch failed.");
        testAssert(worker.Array1 == array1, "Array 1 does not match input.");
        testAssert(worker.Array2 == array2, "Array 2 does not match input.");
        worker.Reset();

        int lval{ 42 };
        int rval{ 20 };
        testAssert(Dispatcher::Execute(array1, array2, paramTester, lval, std::move(rval)),
          "Parameter forwarding dispatch failed.");
        testAssert(paramTester.Success, "Parameter forwarding failed.");
        paramTester.Reset();

        testAssert(Dispatcher::Execute(array1, array2, ForwardedFunctor{}),
          "Functor forwarding dispatch failed.");
        testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
        ForwardedFunctorCalled = false;
      }
      else
      {
        testAssert(!Dispatcher::Execute(array1, array2, worker), "Dispatch should have failed.");
        testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
        testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
        worker.Reset();
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch2BySameValueType()
{
  int errors = 0;

  // Restrictions:
  // - Types must match
  // - Only integral types.
  using Dispatcher = svtkArrayDispatch::Dispatch2BySameValueType<svtkArrayDispatch::Integrals>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = isIntegral(array1->GetDataType());

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = svtkDataTypesCompare(array1->GetDataType(), array2->GetDataType()) != 0;

      if (array1Valid && array2Valid)
      {
        testAssert(Dispatcher::Execute(array1, array2, worker), "Dispatch failed.");
        testAssert(worker.Array1 == array1, "Array 1 does not match input.");
        testAssert(worker.Array2 == array2, "Array 2 does not match input.");
        worker.Reset();

        int lval{ 42 };
        int rval{ 20 };
        testAssert(Dispatcher::Execute(array1, array2, paramTester, lval, std::move(rval)),
          "Parameter forwarding dispatch failed.");
        testAssert(paramTester.Success, "Parameter forwarding failed.");
        paramTester.Reset();

        testAssert(Dispatcher::Execute(array1, array2, ForwardedFunctor{}),
          "Functor forwarding dispatch failed.");
        testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
        ForwardedFunctorCalled = false;
      }
      else
      {
        testAssert(!Dispatcher::Execute(array1, array2, worker), "Dispatch should have failed.");
        testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
        testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
        worker.Reset();
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch3ByArray()
{
  int errors = 0;

  // Restrictions:
  // Array1: SoA
  // Array2: AoS
  // Array3: AoS/SoA float arrays
  using Dispatcher = svtkArrayDispatch::Dispatch3ByArray<SoAArrayList, AoSArrayList,
    svtkTypeList::Create<svtkAOSDataArrayTemplate<float>, svtkSOADataArrayTemplate<float> > >;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = array1->GetArrayType() == svtkAbstractArray::SoADataArrayTemplate;

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = array2->GetArrayType() == svtkAbstractArray::AoSDataArrayTemplate;

      for (svtkDataArray* array3 : Arrays::allArrays)
      {
        bool array3Valid = array3->GetDataType() == SVTK_FLOAT;

        if (array1Valid && array2Valid && array3Valid)
        {
          testAssert(Dispatcher::Execute(array1, array2, array3, worker), "Dispatch failed.");
          testAssert(worker.Array1 == array1, "Array 1 does not match input.");
          testAssert(worker.Array2 == array2, "Array 2 does not match input.");
          testAssert(worker.Array3 == array3, "Array 3 does not match input.");
          worker.Reset();

          int lval{ 42 };
          int rval{ 20 };
          testAssert(
            Dispatcher::Execute(array1, array2, array3, paramTester, lval, std::move(rval)),
            "Parameter forwarding dispatch failed.");
          testAssert(paramTester.Success, "Parameter forwarding failed.");
          paramTester.Reset();

          testAssert(Dispatcher::Execute(array1, array2, array3, ForwardedFunctor{}),
            "Functor forwarding dispatch failed.");
          testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
          ForwardedFunctorCalled = false;
        }
        else
        {
          testAssert(
            !Dispatcher::Execute(array1, array2, array3, worker), "Dispatch should have failed.");
          testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
          testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
          testAssert(worker.Array3 == nullptr, "Array 3 should be nullptr.");
          worker.Reset();
        }
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch3ByValueType()
{
  int errors = 0;

  // Restrictions:
  // Array1: Must be real type.
  // Array2: Must be integer type.
  // Array3: Must be unsigned char type.
  using Dispatcher = svtkArrayDispatch::Dispatch3ByValueType<svtkArrayDispatch::Reals,
    svtkArrayDispatch::Integrals, svtkTypeList::Create<unsigned char> >;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = isReal(array1->GetDataType());

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = isIntegral(array2->GetDataType());

      for (svtkDataArray* array3 : Arrays::allArrays)
      {
        bool array3Valid = svtkDataTypesCompare(array3->GetDataType(), SVTK_UNSIGNED_CHAR) != 0;

        if (array1Valid && array2Valid && array3Valid)
        {
          testAssert(Dispatcher::Execute(array1, array2, array3, worker), "Dispatch failed.");
          testAssert(worker.Array1 == array1, "Array 1 does not match input.");
          testAssert(worker.Array2 == array2, "Array 2 does not match input.");
          testAssert(worker.Array3 == array3, "Array 3 does not match input.");
          worker.Reset();

          int lval{ 42 };
          int rval{ 20 };
          testAssert(
            Dispatcher::Execute(array1, array2, array3, paramTester, lval, std::move(rval)),
            "Parameter forwarding dispatch failed.");
          testAssert(paramTester.Success, "Parameter forwarding failed.");
          paramTester.Reset();

          testAssert(Dispatcher::Execute(array1, array2, array3, ForwardedFunctor{}),
            "Functor forwarding dispatch failed.");
          testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
          ForwardedFunctorCalled = false;
        }
        else
        {
          testAssert(
            !Dispatcher::Execute(array1, array2, array3, worker), "Dispatch should have failed.");
          testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
          testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
          testAssert(worker.Array3 == nullptr, "Array 3 should be nullptr.");
          worker.Reset();
        }
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch3ByArrayWithSameValueType()
{
  int errors = 0;

  // Restrictions:
  // - Array1: SoA
  // - Array2: AoS
  // - Array3: Any array type
  // - All arrays have same ValueType
  using Dispatcher =
    svtkArrayDispatch::Dispatch3ByArrayWithSameValueType<SoAArrayList, AoSArrayList, AllArrayList>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = array1->GetArrayType() == svtkAbstractArray::SoADataArrayTemplate;

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = array2->GetArrayType() == svtkAbstractArray::AoSDataArrayTemplate &&
        svtkDataTypesCompare(array1->GetDataType(), array2->GetDataType());

      for (svtkDataArray* array3 : Arrays::allArrays)
      {
        bool array3Valid = svtkDataTypesCompare(array1->GetDataType(), array3->GetDataType()) != 0;

        if (array1Valid && array2Valid && array3Valid)
        {
          testAssert(Dispatcher::Execute(array1, array2, array3, worker), "Dispatch failed.");
          testAssert(worker.Array1 == array1, "Array 1 does not match input.");
          testAssert(worker.Array2 == array2, "Array 2 does not match input.");
          testAssert(worker.Array3 == array3, "Array 3 does not match input.");
          worker.Reset();

          int lval{ 42 };
          int rval{ 20 };
          testAssert(
            Dispatcher::Execute(array1, array2, array3, paramTester, lval, std::move(rval)),
            "Parameter forwarding dispatch failed.");
          testAssert(paramTester.Success, "Parameter forwarding failed.");
          paramTester.Reset();

          testAssert(Dispatcher::Execute(array1, array2, array3, ForwardedFunctor{}),
            "Functor forwarding dispatch failed.");
          testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
          ForwardedFunctorCalled = false;
        }
        else
        {
          testAssert(
            !Dispatcher::Execute(array1, array2, array3, worker), "Dispatch should have failed.");
          testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
          testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
          testAssert(worker.Array3 == nullptr, "Array 3 should be nullptr.");
          worker.Reset();
        }
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
int TestDispatch3BySameValueType()
{
  int errors = 0;

  // Restrictions:
  // - All arrays must have same ValueType
  // - ValueType must be float, double, or unsigned char.
  using Dispatcher = svtkArrayDispatch::Dispatch3BySameValueType<
    svtkTypeList::Append<svtkArrayDispatch::Reals, unsigned char>::Result>;
  TestWorker worker;
  ForwardedParams paramTester;

  for (svtkDataArray* array1 : Arrays::allArrays)
  {
    bool array1Valid = isReal(array1->GetDataType()) ||
      svtkDataTypesCompare(array1->GetDataType(), SVTK_UNSIGNED_CHAR);

    for (svtkDataArray* array2 : Arrays::allArrays)
    {
      bool array2Valid = svtkDataTypesCompare(array1->GetDataType(), array2->GetDataType()) != 0;

      for (svtkDataArray* array3 : Arrays::allArrays)
      {
        bool array3Valid = svtkDataTypesCompare(array1->GetDataType(), array3->GetDataType()) != 0;

        if (array1Valid && array2Valid && array3Valid)
        {
          testAssert(Dispatcher::Execute(array1, array2, array3, worker), "Dispatch failed.");
          testAssert(worker.Array1 == array1, "Array 1 does not match input.");
          testAssert(worker.Array2 == array2, "Array 2 does not match input.");
          testAssert(worker.Array3 == array3, "Array 3 does not match input.");
          worker.Reset();

          int lval{ 42 };
          int rval{ 20 };
          testAssert(
            Dispatcher::Execute(array1, array2, array3, paramTester, lval, std::move(rval)),
            "Parameter forwarding dispatch failed.");
          testAssert(paramTester.Success, "Parameter forwarding failed.");
          paramTester.Reset();

          testAssert(Dispatcher::Execute(array1, array2, array3, ForwardedFunctor{}),
            "Functor forwarding dispatch failed.");
          testAssert(ForwardedFunctorCalled, "Functor forwarding failed.");
          ForwardedFunctorCalled = false;
        }
        else
        {
          testAssert(
            !Dispatcher::Execute(array1, array2, array3, worker), "Dispatch should have failed.");
          testAssert(worker.Array1 == nullptr, "Array 1 should be nullptr.");
          testAssert(worker.Array2 == nullptr, "Array 2 should be nullptr.");
          testAssert(worker.Array3 == nullptr, "Array 3 should be nullptr.");
          worker.Reset();
        }
      }
    }
  }

  return errors;
}

//------------------------------------------------------------------------------
Arrays::Arrays()
{
  aosDouble = svtkAOSDataArrayTemplate<double>::New();
  aosFloat = svtkAOSDataArrayTemplate<float>::New();
  aosInt = svtkAOSDataArrayTemplate<int>::New();
  aosUnsignedChar = svtkAOSDataArrayTemplate<unsigned char>::New();
  aosIdType = svtkAOSDataArrayTemplate<svtkIdType>::New();

  soaDouble = svtkSOADataArrayTemplate<double>::New();
  soaFloat = svtkSOADataArrayTemplate<float>::New();
  soaInt = svtkSOADataArrayTemplate<int>::New();
  soaUnsignedChar = svtkSOADataArrayTemplate<unsigned char>::New();
  soaIdType = svtkSOADataArrayTemplate<svtkIdType>::New();

  aosArrays.push_back(aosDouble);
  aosArrays.push_back(aosFloat);
  aosArrays.push_back(aosInt);
  aosArrays.push_back(aosUnsignedChar);
  aosArrays.push_back(aosIdType);

  soaArrays.push_back(soaDouble);
  soaArrays.push_back(soaFloat);
  soaArrays.push_back(soaInt);
  soaArrays.push_back(soaUnsignedChar);
  soaArrays.push_back(soaIdType);

  std::copy(aosArrays.begin(), aosArrays.end(), std::back_inserter(allArrays));
  std::copy(soaArrays.begin(), soaArrays.end(), std::back_inserter(allArrays));
}

//------------------------------------------------------------------------------
Arrays::~Arrays()
{
  aosDouble->Delete();
  aosFloat->Delete();
  aosInt->Delete();
  aosUnsignedChar->Delete();
  aosIdType->Delete();
  soaDouble->Delete();
  soaFloat->Delete();
  soaInt->Delete();
  soaUnsignedChar->Delete();
  soaIdType->Delete();
}

} // end anon namespace

//------------------------------------------------------------------------------
int TestArrayDispatchers(int, char*[])
{
  int errors = 0;
  Arrays arrays;
  (void)arrays; // unused, just manages static memory

  errors += TestDispatch();
  errors += TestDispatchByArray();
  errors += TestDispatchByValueType();
  errors += TestDispatch2ByArray();
  errors += TestDispatch2ByValueType();
  errors += TestDispatch2ByArrayWithSameValueType();
  errors += TestDispatch2BySameValueType();
  errors += TestDispatch3ByArray();
  errors += TestDispatch3ByValueType();
  errors += TestDispatch3ByArrayWithSameValueType();
  errors += TestDispatch3BySameValueType();

  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

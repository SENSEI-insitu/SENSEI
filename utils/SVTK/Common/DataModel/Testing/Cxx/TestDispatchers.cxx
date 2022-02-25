/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestDispatchers.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME Test Dispatchers
// .SECTION Description
// Tests svtkDispatcher and svtkDoubleDispatcher

#include "svtkDataArrayDispatcher.h"
#include "svtkDispatcher.h"
#include "svtkDoubleDispatcher.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"

// classes we will be using in the test
#include "svtkCharArray.h"
#include "svtkDoubleArray.h"
#include "svtkIntArray.h"
#include "svtkPoints.h"
#include "svtkStringArray.h"

#include <algorithm>
#include <stdexcept>
namespace
{

void test_expression(bool valid, const std::string& msg)
{
  if (!valid)
  {
    throw std::runtime_error(msg);
  }
}

template <typename T, typename U>
inline T* as(U* u)
{
  return dynamic_cast<T*>(u);
}

struct singleFunctor
{
  int timesCalled;
  singleFunctor()
    : timesCalled(0)
  {
  }
  template <typename T>
  int operator()(T&)
  {
    return ++timesCalled;
  }
};

struct doubleFunctor
{
  int timesCalled;
  doubleFunctor()
    : timesCalled(0)
  {
  }
  template <typename T, typename U>
  int operator()(T&, U&)
  {
    return ++timesCalled;
  }
};

// type traits for svtkTTFunctor and pointsWrapper
template <typename T>
struct FieldType;
template <>
struct FieldType<svtkIntArray>
{
  enum
  {
    SVTK_DATA_TYPE = SVTK_INT
  };
  typedef int ValueType;
};

template <>
struct FieldType<svtkDoubleArray>
{
  enum
  {
    SVTK_DATA_TYPE = SVTK_DOUBLE
  };
  typedef double ValueType;
};

template <>
struct FieldType<svtkCharArray>
{
  enum
  {
    SVTK_DATA_TYPE = SVTK_CHAR
  };
  typedef char ValueType;
};

// this functor replaces the usage of SVTK_TT macro, by showing
// how to use template traits
struct svtkTTFunctor
{
  template <typename T>
  void operator()(T& t) const
  {
    // example that sorts, only works on single component
    typedef typename FieldType<T>::ValueType ValueType;
    if (t.GetNumberOfComponents() == 1)
    {
      ValueType* start = static_cast<ValueType*>(t.GetVoidPointer(0));
      ValueType* end = static_cast<ValueType*>(t.GetVoidPointer(t.GetNumberOfTuples()));
      std::sort(start, end);
    }
  }
};

struct pointsFunctor
{
  svtkPoints* operator()(svtkDoubleArray& dataArray) const
  {
    svtkPoints* points = svtkPoints::New();
    points->SetData(&dataArray);
    return points;
  }

  svtkPoints* operator()(svtkIntArray& dataArray) const
  {
    svtkPoints* points = svtkPoints::New();
    points->SetNumberOfPoints(dataArray.GetNumberOfTuples());
    return points;
  }
};

}

bool TestSingleDispatch()
{
  // statefull dispatching
  singleFunctor functor;
  svtkDispatcher<svtkObject, int> dispatcher;
  dispatcher.Add<svtkDoubleArray>(&functor);
  dispatcher.Add<svtkStringArray>(&functor);
  dispatcher.Add<svtkIntArray>(&functor);

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkStringArray> stringArray;
  svtkNew<svtkIntArray> intArray;
  svtkNew<svtkPoints> pointsArray;

  int result = dispatcher.Go(as<svtkObject>(doubleArray.GetPointer()));
  test_expression(result == 1, "double array dispatch failed with statefull functor");

  result = dispatcher.Go(stringArray.GetPointer());
  test_expression(result == 2, "string array dispatch failed with statefull functor");

  result = dispatcher.Go(intArray.GetPointer());
  test_expression(result == 3, "int array dispatch failed with statefull functor");

  result = dispatcher.Go(pointsArray.GetPointer());
  test_expression(result == 0, "points array didn't fail");

  return true;
}

bool TestStatelessSingleDispatch()
{
  // stateless dispatching
  svtkDispatcher<svtkObject, int> dispatcher;
  dispatcher.Add<svtkDoubleArray>(singleFunctor());
  dispatcher.Add<svtkStringArray>(singleFunctor());

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkStringArray> stringArray;

  int result = dispatcher.Go(doubleArray.GetPointer());
  test_expression(result == 1, "double array dispatch failed with stateless functor");

  result = dispatcher.Go(as<svtkObject>(stringArray.GetPointer()));
  test_expression(result == 1, "string array dispatch failed with stateless functor");

  return true;
}

bool TestDoubleDispatch()
{
  // statefull dispatching
  doubleFunctor functor;
  svtkDoubleDispatcher<svtkObject, svtkObject, int> dispatcher;
  dispatcher.Add<svtkDoubleArray, svtkStringArray>(&functor);
  dispatcher.Add<svtkStringArray, svtkStringArray>(&functor);
  dispatcher.Add<svtkIntArray, svtkDoubleArray>(&functor);

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkStringArray> stringArray;
  svtkNew<svtkIntArray> intArray;
  svtkNew<svtkPoints> pointsArray;

  int result =
    dispatcher.Go(as<svtkObject>(doubleArray.GetPointer()), as<svtkObject>(stringArray.GetPointer()));
  test_expression(result == 1, "double array dispatch failed with statefull functor");

  result = dispatcher.Go(stringArray.GetPointer(), stringArray.GetPointer());
  test_expression(result == 2, "string array dispatch failed with statefull functor");

  result = dispatcher.Go(as<svtkObject>(intArray.GetPointer()), doubleArray.GetPointer());
  test_expression(result == 3, "int array dispatch failed with statefull functor");

  result = dispatcher.Go(intArray.GetPointer(), pointsArray.GetPointer());
  test_expression(result == 0, "points array didn't fail");

  return true;
}

bool TestStatelessDoubleDispatch()
{
  // stateless dispatching
  svtkDoubleDispatcher<svtkObject, svtkObject, int> dispatcher;
  dispatcher.Add<svtkDoubleArray, svtkStringArray>(doubleFunctor());
  dispatcher.Add<svtkStringArray, svtkStringArray>(doubleFunctor());
  dispatcher.Add<svtkIntArray, svtkDoubleArray>(doubleFunctor());

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkStringArray> stringArray;
  svtkNew<svtkIntArray> intArray;
  svtkNew<svtkPoints> pointsArray;

  int result = dispatcher.Go(doubleArray.GetPointer(), stringArray.GetPointer());
  test_expression(result == 1, "double array dispatch failed with statefull functor");

  result = dispatcher.Go(stringArray.GetPointer(), stringArray.GetPointer());
  test_expression(result == 1, "string array dispatch failed with statefull functor");

  result = dispatcher.Go(intArray.GetPointer(), doubleArray.GetPointer());
  test_expression(result == 1, "int array dispatch failed with statefull functor");

  result = dispatcher.Go(intArray.GetPointer(), pointsArray.GetPointer());
  test_expression(result == 0, "points array didn't fail");

  return true;
}

bool TestMixedDispatch()
{
  // stateless dispatching
  singleFunctor functor;
  svtkDispatcher<svtkDataArray, int> dispatcher;
  dispatcher.Add<svtkDoubleArray>(&functor);
  dispatcher.Add<svtkIntArray>(&functor);
  dispatcher.Add<svtkCharArray>(singleFunctor());

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkIntArray> intArray;
  svtkNew<svtkCharArray> charArray;

  int result = dispatcher.Go(as<svtkDataArray>(doubleArray.GetPointer()));
  result = dispatcher.Go(intArray.GetPointer());
  test_expression(result == 2, "unexpected");
  result = dispatcher.Go(doubleArray.GetPointer());
  test_expression(result == 3, "statefull functor failed with int and double");

  result = dispatcher.Go(charArray.GetPointer());
  test_expression(result == 1, "");

  return true;
}

bool TestSVTKTTReplacement()
{
  // stateless dispatching
  svtkDispatcher<svtkDataArray> dispatcher; // default return type is void
  dispatcher.Add<svtkDoubleArray>(svtkTTFunctor());
  dispatcher.Add<svtkIntArray>(svtkTTFunctor());

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  svtkNew<svtkIntArray> intArray;

  doubleArray->SetNumberOfValues(10);
  intArray->SetNumberOfValues(10);

  for (int i = 0; i < 10; ++i)
  {
    doubleArray->SetValue(i, 10 - i);
    intArray->SetValue(i, -10 * i);
  }

  // sort the array, passing in as svtkObject to show we use RTTI
  // to get out the derived class info
  dispatcher.Go(as<svtkDataArray>(doubleArray.GetPointer()));
  dispatcher.Go(as<svtkDataArray>(intArray.GetPointer()));

  // verify the array is sorted, by checking min & max
  test_expression(doubleArray->GetValue(0) == 1, "double array not sorted");
  test_expression(doubleArray->GetValue(9) == 10, "double array not sorted");

  test_expression(intArray->GetValue(0) == -90, "int array not sorted");
  test_expression(intArray->GetValue(9) == 0, "int array not sorted");

  return true;
}

bool TestReturnVtkObject()
{
  // This example shows how to return a svtkObject that is filled by the algorithm
  // that you passed in.
  svtkDispatcher<svtkDataArray, svtkPoints*> dispatcher; // default return type is void
  dispatcher.Add<svtkDoubleArray>(pointsFunctor());
  dispatcher.Add<svtkIntArray>(pointsFunctor());

  // verify the dispatching
  svtkNew<svtkDoubleArray> doubleArray;
  doubleArray->SetNumberOfComponents(3);
  doubleArray->SetNumberOfTuples(1);

  // make sure the result isn't copied anywhere
  svtkPoints* result = dispatcher.Go(as<svtkDataArray>(doubleArray.GetPointer()));

  test_expression(result != nullptr, "Returned points not valid");
  test_expression(result->GetData() == doubleArray.GetPointer(),
    "Returned points not equal to the passed in double array");
  result->Delete();

  // on an integer function we should get a whole new points array
  svtkNew<svtkIntArray> intArray;
  result = dispatcher.Go(as<svtkDataArray>(intArray.GetPointer()));

  test_expression(result != nullptr, "Returned points not valid");
  result->Delete();

  return true;
}

int TestDispatchers(int /*argc*/, char* /*argv*/[])
{

  bool passed = TestSingleDispatch();
  passed &= TestStatelessSingleDispatch();
  passed &= TestDoubleDispatch();
  passed &= TestStatelessDoubleDispatch();
  passed &= TestMixedDispatch();
  passed &= TestSVTKTTReplacement();
  passed &= TestReturnVtkObject();

  return passed ? 0 : 1;
}

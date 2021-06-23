/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayCasting.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <svtkDenseArray.h>
#include <svtkSmartPointer.h>
#include <svtkSparseArray.h>
#include <svtkTryDowncast.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string.hpp>

#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

#define test_expression(expression)                                                                \
  {                                                                                                \
    if (!(expression))                                                                             \
    {                                                                                              \
      std::ostringstream buffer;                                                                   \
      buffer << "Expression failed at line " << __LINE__ << ": " << #expression;                   \
      throw std::runtime_error(buffer.str());                                                      \
    }                                                                                              \
  }

class DowncastTest
{
public:
  DowncastTest(int& count)
    : Count(count)
  {
  }

  template <typename T>
  void operator()(T* svtkNotUsed(array)) const
  {
    ++Count;
  }

  int& Count;

private:
  DowncastTest& operator=(const DowncastTest&);
};

template <template <typename> class TargetT, typename TypesT>
void SuccessTest(svtkObject* source, int line)
{
  int count = 0;
  if (!svtkTryDowncast<TargetT, TypesT>(source, DowncastTest(count)))
  {
    std::ostringstream buffer;
    buffer << "Expression failed at line " << line;
    throw std::runtime_error(buffer.str());
  }

  if (count != 1)
  {
    std::ostringstream buffer;
    buffer << "Functor was called " << count << " times at line " << line;
    throw std::runtime_error(buffer.str());
  }
}

template <template <typename> class TargetT, typename TypesT>
void FailTest(svtkObject* source, int line)
{
  int count = 0;
  if (svtkTryDowncast<TargetT, TypesT>(source, DowncastTest(count)))
  {
    std::ostringstream buffer;
    buffer << "Expression failed at line " << line;
    throw std::runtime_error(buffer.str());
  }

  if (count != 0)
  {
    std::ostringstream buffer;
    buffer << "Functor was called " << count << " times at line " << line;
    throw std::runtime_error(buffer.str());
  }
}

/*
// This functor increments array values in-place using a parameter passed via the algorithm (instead
of a parameter
// stored in the functor).  It can work with any numeric array type.
struct IncrementValues
{
  template<typename T>
  void operator()(T* array, int amount) const
  {
    for(svtkIdType n = 0; n != array->GetNonNullSize(); ++n)
      array->SetValueN(n, array->GetValueN(n) + amount);
  }
};

// This functor converts strings in-place to a form suitable for case-insensitive comparison.  It's
an example of
// how you can write generic code while still specializing functionality on a case-by-case basis,
since
// in this situation we want to use some special functionality provided by svtkUnicodeString.
struct FoldCase
{
  template<typename ValueT>
  void operator()(svtkTypedArray<ValueT>* array) const
  {
    for(svtkIdType n = 0; n != array->GetNonNullSize(); ++n)
      {
      ValueT value = array->GetValueN(n);
      boost::algorithm::to_lower(value);
      array->SetValueN(n, value);
      }
  }

  void operator()(svtkTypedArray<svtkUnicodeString>* array) const
  {
    for(svtkIdType n = 0; n != array->GetNonNullSize(); ++n)
      array->SetValueN(n, array->GetValueN(n).fold_case());
  }
};

// This functor efficiently creates a transposed array.  It's one example of how you can create an
output array
// with the same type as an input array.
struct Transpose
{
  Transpose(svtkSmartPointer<svtkArray>& result_matrix) : ResultMatrix(result_matrix) {}

  template<typename ValueT>
  void operator()(svtkDenseArray<ValueT>* input) const
  {
    if(input->GetDimensions() != 2 || input->GetExtents()[0] != input->GetExtents()[1])
      throw std::runtime_error("A square matrix is required.");

    svtkDenseArray<ValueT>* output = svtkDenseArray<ValueT>::SafeDownCast(input->DeepCopy());
    for(svtkIdType i = 0; i != input->GetExtents()[0]; ++i)
      {
      for(svtkIdType j = i + 1; j != input->GetExtents()[1]; ++j)
        {
        output->SetValue(i, j, input->GetValue(j, i));
        output->SetValue(j, i, input->GetValue(i, j));
        }
      }

    this->ResultMatrix = output;
  }

  svtkSmartPointer<svtkArray>& ResultMatrix;
};
*/

//
//
// Here are some examples of how the algorithm might be called.
//
//

int TestArrayCasting(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    SVTK_CREATE(svtkDenseArray<int>, dense_int);
    SVTK_CREATE(svtkDenseArray<double>, dense_double);
    SVTK_CREATE(svtkDenseArray<svtkStdString>, dense_string);
    SVTK_CREATE(svtkSparseArray<int>, sparse_int);
    SVTK_CREATE(svtkSparseArray<double>, sparse_double);
    SVTK_CREATE(svtkSparseArray<svtkStdString>, sparse_string);

    SuccessTest<svtkTypedArray, svtkIntegerTypes>(dense_int, __LINE__);
    FailTest<svtkTypedArray, svtkIntegerTypes>(dense_double, __LINE__);
    FailTest<svtkTypedArray, svtkIntegerTypes>(dense_string, __LINE__);
    SuccessTest<svtkTypedArray, svtkIntegerTypes>(sparse_int, __LINE__);
    FailTest<svtkTypedArray, svtkIntegerTypes>(sparse_double, __LINE__);
    FailTest<svtkTypedArray, svtkIntegerTypes>(sparse_string, __LINE__);

    FailTest<svtkTypedArray, svtkFloatingPointTypes>(dense_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkFloatingPointTypes>(dense_double, __LINE__);
    FailTest<svtkTypedArray, svtkFloatingPointTypes>(dense_string, __LINE__);
    FailTest<svtkTypedArray, svtkFloatingPointTypes>(sparse_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkFloatingPointTypes>(sparse_double, __LINE__);
    FailTest<svtkTypedArray, svtkFloatingPointTypes>(sparse_string, __LINE__);

    SuccessTest<svtkTypedArray, svtkNumericTypes>(dense_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkNumericTypes>(dense_double, __LINE__);
    FailTest<svtkTypedArray, svtkNumericTypes>(dense_string, __LINE__);
    SuccessTest<svtkTypedArray, svtkNumericTypes>(sparse_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkNumericTypes>(sparse_double, __LINE__);
    FailTest<svtkTypedArray, svtkNumericTypes>(sparse_string, __LINE__);

    FailTest<svtkTypedArray, svtkStringTypes>(dense_int, __LINE__);
    FailTest<svtkTypedArray, svtkStringTypes>(dense_double, __LINE__);
    SuccessTest<svtkTypedArray, svtkStringTypes>(dense_string, __LINE__);
    FailTest<svtkTypedArray, svtkStringTypes>(sparse_int, __LINE__);
    FailTest<svtkTypedArray, svtkStringTypes>(sparse_double, __LINE__);
    SuccessTest<svtkTypedArray, svtkStringTypes>(sparse_string, __LINE__);

    SuccessTest<svtkTypedArray, svtkAllTypes>(dense_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkAllTypes>(dense_double, __LINE__);
    SuccessTest<svtkTypedArray, svtkAllTypes>(dense_string, __LINE__);
    SuccessTest<svtkTypedArray, svtkAllTypes>(sparse_int, __LINE__);
    SuccessTest<svtkTypedArray, svtkAllTypes>(sparse_double, __LINE__);
    SuccessTest<svtkTypedArray, svtkAllTypes>(sparse_string, __LINE__);

    SuccessTest<svtkDenseArray, svtkAllTypes>(dense_int, __LINE__);
    FailTest<svtkDenseArray, svtkAllTypes>(sparse_int, __LINE__);
    FailTest<svtkSparseArray, svtkAllTypes>(dense_int, __LINE__);
    SuccessTest<svtkSparseArray, svtkAllTypes>(sparse_int, __LINE__);

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}

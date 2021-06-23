/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayNullValues.cxx

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

#include <svtkSmartPointer.h>
#include <svtkSparseArray.h>

#include <iostream>
#include <stdexcept>

template <typename T>
void VerifyType(const T& DefaultNull, const T& AlternateNull)
{
  // Create a sparse array ...
  svtkSmartPointer<svtkSparseArray<T> > array = svtkSmartPointer<svtkSparseArray<T> >::New();
  array->Resize(2);

  // Verify that the default nullptr value is iniitialized correctly ...
  if (array->GetNullValue() != DefaultNull)
  {
    throw std::runtime_error(
      "Incorrect default nullptr value for " + std::string(array->GetClassName()));
  }

  // Verify that GetValue() returns the default nullptr value for nullptr elements ...
  if (array->GetValue(1) != DefaultNull)
  {
    throw std::runtime_error(
      "Empty value did not return default nullptr for " + std::string(array->GetClassName()));
  }

  // Verify that we can override the default nullptr value ...
  array->SetNullValue(AlternateNull);
  if (array->GetNullValue() != AlternateNull)
  {
    throw std::runtime_error(
      "Error overriding nullptr value for " + std::string(array->GetClassName()));
  }

  // Verify that GetValue() returns the alternate nullptr value forr nullptr elements ...
  if (array->GetValue(1) != AlternateNull)
  {
    throw std::runtime_error(
      "Empty value did not overridden nullptr for " + std::string(array->GetClassName()));
  }
}

int TestArrayNullValues(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    VerifyType<svtkTypeInt8>(0, 1);
    VerifyType<svtkTypeUInt8>(0, 1);
    VerifyType<svtkTypeInt16>(0, 1);
    VerifyType<svtkTypeUInt16>(0, 1);
    VerifyType<svtkTypeInt32>(0, 1);
    VerifyType<svtkTypeUInt32>(0, 1);
    VerifyType<svtkTypeFloat32>(0.0f, 1);
    VerifyType<svtkTypeFloat64>(0.0, 1);
    VerifyType<svtkIdType>(0, 1);
    VerifyType<svtkStdString>(svtkStdString(""), svtkStdString("foo"));

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}

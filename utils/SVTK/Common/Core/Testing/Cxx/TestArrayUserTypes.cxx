/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayUserTypes.cxx

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

#include <iostream>
#include <sstream>
#include <stdexcept>

#define test_expression(expression)                                                                \
  {                                                                                                \
    if (!(expression))                                                                             \
    {                                                                                              \
      std::ostringstream buffer;                                                                   \
      buffer << "Expression failed at line " << __LINE__ << ": " << #expression;                   \
      throw std::runtime_error(buffer.str());                                                      \
    }                                                                                              \
  }

class UserType
{
public:
  UserType()
    : Value("")
  {
  }

  UserType(const svtkStdString& value)
    : Value(value)
  {
  }

  bool operator==(const UserType& other) const { return this->Value == other.Value; }

  svtkStdString Value;
};

template <>
inline UserType svtkVariantCast<UserType>(const svtkVariant& value, bool* valid)
{
  if (valid)
    *valid = true;
  return UserType(value.ToString());
}

template <>
inline svtkVariant svtkVariantCreate<UserType>(const UserType& value)
{
  return svtkVariant(value.Value);
}

int TestArrayUserTypes(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    svtkSmartPointer<svtkDenseArray<UserType> > dense =
      svtkSmartPointer<svtkDenseArray<UserType> >::New();
    dense->Resize(3, 4);
    dense->Fill(UserType("red"));
    for (svtkArray::SizeT n = 0; n != dense->GetNonNullSize(); ++n)
    {
      test_expression(dense->GetValueN(n) == UserType("red"));
    }

    dense->SetValue(1, 2, UserType("green"));
    test_expression(dense->GetValue(1, 2) == UserType("green"));

    dense->SetVariantValue(1, 2, svtkVariant("puce"));
    test_expression(dense->GetValue(1, 2) == UserType("puce"));
    test_expression(dense->GetVariantValue(1, 2) == svtkVariant("puce"));

    svtkSmartPointer<svtkSparseArray<UserType> > sparse =
      svtkSmartPointer<svtkSparseArray<UserType> >::New();
    sparse->Resize(3, 4);
    sparse->SetNullValue(UserType("blue"));
    test_expression(sparse->GetNullValue() == UserType("blue"));
    test_expression(sparse->GetValue(1, 2) == UserType("blue"));

    sparse->SetValue(0, 1, UserType("white"));
    test_expression(sparse->GetValue(0, 1) == UserType("white"));

    sparse->AddValue(2, 3, UserType("yellow"));
    test_expression(sparse->GetValue(2, 3) == UserType("yellow"));

    sparse->SetVariantValue(2, 3, svtkVariant("slate"));
    test_expression(sparse->GetValue(2, 3) == UserType("slate"));
    test_expression(sparse->GetVariantValue(2, 3) == svtkVariant("slate"));

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}

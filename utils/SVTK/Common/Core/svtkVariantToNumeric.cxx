/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantToNumeric.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/

//----------------------------------------------------------------------------
// Templated definition of ToNumeric, isolated into its very own file to
// allow it to be defined before its use with most compilers, but after its
// use for (at least) Visual Studio 6.

template <typename T>
T svtkVariant::ToNumeric(bool* valid, T* svtkNotUsed(ignored)) const
{
  if (valid)
  {
    *valid = true;
  }
  if (this->IsString())
  {
    return svtkVariantStringToNumeric<T>(*this->Data.String, valid);
  }
  if (this->IsFloat())
  {
    return static_cast<T>(this->Data.Float);
  }
  if (this->IsDouble())
  {
    return static_cast<T>(this->Data.Double);
  }
  if (this->IsChar())
  {
    return static_cast<T>(this->Data.Char);
  }
  if (this->IsUnsignedChar())
  {
    return static_cast<T>(this->Data.UnsignedChar);
  }
  if (this->IsSignedChar())
  {
    return static_cast<T>(this->Data.SignedChar);
  }
  if (this->IsShort())
  {
    return static_cast<T>(this->Data.Short);
  }
  if (this->IsUnsignedShort())
  {
    return static_cast<T>(this->Data.UnsignedShort);
  }
  if (this->IsInt())
  {
    return static_cast<T>(this->Data.Int);
  }
  if (this->IsUnsignedInt())
  {
    return static_cast<T>(this->Data.UnsignedInt);
  }
  if (this->IsLong())
  {
    return static_cast<T>(this->Data.Long);
  }
  if (this->IsUnsignedLong())
  {
    return static_cast<T>(this->Data.UnsignedLong);
  }
  if (this->IsLongLong())
  {
    return static_cast<T>(this->Data.LongLong);
  }
  if (this->IsUnsignedLongLong())
  {
    return static_cast<T>(this->Data.UnsignedLongLong);
  }
  // For arrays, convert the first value to the appropriate type.
  if (this->IsArray())
  {
    if (this->Data.SVTKObject->IsA("svtkDataArray"))
    {
      // Note: This are not the best conversion.
      //       We covert the first value to double, then
      //       cast it back to the appropriate numeric type.
      svtkDataArray* da = svtkDataArray::SafeDownCast(this->Data.SVTKObject);
      return static_cast<T>(da->GetTuple1(0));
    }
    if (this->Data.SVTKObject->IsA("svtkVariantArray"))
    {
      // Note: This are not the best conversion.
      //       We covert the first value to double, then
      //       cast it back to the appropriate numeric type.
      svtkVariantArray* va = svtkVariantArray::SafeDownCast(this->Data.SVTKObject);
      return static_cast<T>(va->GetValue(0).ToDouble());
    }
    if (this->Data.SVTKObject->IsA("svtkStringArray"))
    {
      svtkStringArray* sa = svtkStringArray::SafeDownCast(this->Data.SVTKObject);
      return svtkVariantStringToNumeric<T>(sa->GetValue(0), valid);
    }
  }
  if (valid)
  {
    *valid = false;
  }
  return static_cast<T>(0);
}

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantCast.h

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

/**
 * @class   svtkVariantCast
 *
 * Converts a svtkVariant to some other type.  Wherever possible, implicit conversions are
 * performed, so this method can be used to convert from nearly any type to a string, or
 * from a string to nearly any type.  Note that some conversions may fail at runtime, such
 * as a conversion from the string "abc" to a numeric type.
 *
 * The optional 'valid' flag can be used by callers to verify whether conversion succeeded.
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkVariantCast_h
#define svtkVariantCast_h

#include "svtkUnicodeString.h"
#include <typeinfo> // for warnings

template <typename T>
T svtkVariantCast(const svtkVariant& value, bool* valid = nullptr)
{
  svtkGenericWarningMacro(<< "Cannot convert svtkVariant containing [" << value.GetTypeAsString()
                         << "] "
                         << "to unsupported type [" << typeid(T).name() << "].  "
                         << "Create a svtkVariantCast<> specialization to eliminate this warning.");

  if (valid)
    *valid = false;

  static T dummy;
  return dummy;
}

template <>
inline char svtkVariantCast<char>(const svtkVariant& value, bool* valid)
{
  return value.ToChar(valid);
}

template <>
inline signed char svtkVariantCast<signed char>(const svtkVariant& value, bool* valid)
{
  return value.ToSignedChar(valid);
}

template <>
inline unsigned char svtkVariantCast<unsigned char>(const svtkVariant& value, bool* valid)
{
  return value.ToUnsignedChar(valid);
}

template <>
inline short svtkVariantCast<short>(const svtkVariant& value, bool* valid)
{
  return value.ToShort(valid);
}

template <>
inline unsigned short svtkVariantCast<unsigned short>(const svtkVariant& value, bool* valid)
{
  return value.ToUnsignedShort(valid);
}

template <>
inline int svtkVariantCast<int>(const svtkVariant& value, bool* valid)
{
  return value.ToInt(valid);
}

template <>
inline unsigned int svtkVariantCast<unsigned int>(const svtkVariant& value, bool* valid)
{
  return value.ToUnsignedInt(valid);
}

template <>
inline long svtkVariantCast<long>(const svtkVariant& value, bool* valid)
{
  return value.ToLong(valid);
}

template <>
inline unsigned long svtkVariantCast<unsigned long>(const svtkVariant& value, bool* valid)
{
  return value.ToUnsignedLong(valid);
}

template <>
inline long long svtkVariantCast<long long>(const svtkVariant& value, bool* valid)
{
  return value.ToLongLong(valid);
}

template <>
inline unsigned long long svtkVariantCast<unsigned long long>(const svtkVariant& value, bool* valid)
{
  return value.ToUnsignedLongLong(valid);
}

template <>
inline float svtkVariantCast<float>(const svtkVariant& value, bool* valid)
{
  return value.ToFloat(valid);
}

template <>
inline double svtkVariantCast<double>(const svtkVariant& value, bool* valid)
{
  return value.ToDouble(valid);
}

template <>
inline svtkStdString svtkVariantCast<svtkStdString>(const svtkVariant& value, bool* valid)
{
  if (valid)
    *valid = true;

  return value.ToString();
}

template <>
inline svtkUnicodeString svtkVariantCast<svtkUnicodeString>(const svtkVariant& value, bool* valid)
{
  if (valid)
    *valid = true;

  return value.ToUnicodeString();
}

template <>
inline svtkVariant svtkVariantCast<svtkVariant>(const svtkVariant& value, bool* valid)
{
  if (valid)
    *valid = true;

  return value;
}

#endif

// SVTK-HeaderTest-Exclude: svtkVariantCast.h

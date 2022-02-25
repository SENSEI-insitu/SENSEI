/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantExtract.h

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
 * @class   svtkVariantExtract
 *
 * Performs an explicit conversion from a svtkVariant to the type that it contains.  Implicit
 * conversions are *not* performed, so casting a svtkVariant containing one type (e.g. double)
 * to a different type (e.g. string) will not convert between types.
 *
 * Callers should use the 'valid' flag to verify whether the extraction succeeded.
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkVariantExtract_h
#define svtkVariantExtract_h

#include <typeinfo> // for typeid

template <typename T>
T svtkVariantExtract(const svtkVariant& value, bool& valid)
{
  svtkGenericWarningMacro(
    << "Cannot convert svtkVariant containing [" << value.GetTypeAsString() << "] "
    << "to unsupported type [" << typeid(T).name() << "].  "
    << "Create a svtkVariantExtract<> specialization to eliminate this warning.");

  valid = false;

  static T dummy;
  return dummy;
}

template <>
inline char svtkVariantExtract<char>(const svtkVariant& value, bool& valid)
{
  valid = value.IsChar();
  return valid ? value.ToChar() : 0;
}

template <>
inline unsigned char svtkVariantExtract<unsigned char>(const svtkVariant& value, bool& valid)
{
  valid = value.IsUnsignedChar();
  return valid ? value.ToUnsignedChar() : 0;
}

template <>
inline short svtkVariantExtract<short>(const svtkVariant& value, bool& valid)
{
  valid = value.IsShort();
  return valid ? value.ToShort() : 0;
}

template <>
inline unsigned short svtkVariantExtract<unsigned short>(const svtkVariant& value, bool& valid)
{
  valid = value.IsUnsignedShort();
  return valid ? value.ToUnsignedShort() : 0;
}

template <>
inline int svtkVariantExtract<int>(const svtkVariant& value, bool& valid)
{
  valid = value.IsInt();
  return valid ? value.ToInt() : 0;
}

template <>
inline unsigned int svtkVariantExtract<unsigned int>(const svtkVariant& value, bool& valid)
{
  valid = value.IsUnsignedInt();
  return valid ? value.ToUnsignedInt() : 0;
}

template <>
inline long svtkVariantExtract<long>(const svtkVariant& value, bool& valid)
{
  valid = value.IsLong();
  return valid ? value.ToLong() : 0;
}

template <>
inline unsigned long svtkVariantExtract<unsigned long>(const svtkVariant& value, bool& valid)
{
  valid = value.IsUnsignedLong();
  return valid ? value.ToUnsignedLong() : 0;
}

template <>
inline long long svtkVariantExtract<long long>(const svtkVariant& value, bool& valid)
{
  valid = value.IsLongLong();
  return valid ? value.ToLongLong() : 0;
}

template <>
inline unsigned long long svtkVariantExtract<unsigned long long>(
  const svtkVariant& value, bool& valid)
{
  valid = value.IsUnsignedLongLong();
  return valid ? value.ToUnsignedLongLong() : 0;
}

template <>
inline float svtkVariantExtract<float>(const svtkVariant& value, bool& valid)
{
  valid = value.IsFloat();
  return valid ? value.ToFloat() : 0.0f;
}

template <>
inline double svtkVariantExtract<double>(const svtkVariant& value, bool& valid)
{
  valid = value.IsDouble();
  return valid ? value.ToDouble() : 0.0;
}

template <>
inline svtkStdString svtkVariantExtract<svtkStdString>(const svtkVariant& value, bool& valid)
{
  valid = value.IsString();
  return valid ? value.ToString() : svtkStdString();
}

template <>
inline svtkUnicodeString svtkVariantExtract<svtkUnicodeString>(const svtkVariant& value, bool& valid)
{
  valid = value.IsUnicodeString();
  return valid ? value.ToUnicodeString() : svtkUnicodeString();
}

template <>
inline svtkVariant svtkVariantExtract<svtkVariant>(const svtkVariant& value, bool& valid)
{
  valid = true;
  return value;
}

#endif

// SVTK-HeaderTest-Exclude: svtkVariantExtract.h

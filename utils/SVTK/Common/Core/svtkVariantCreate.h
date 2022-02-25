/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantCreate.h

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
 * @class   svtkVariantCreate
 *
 * Performs an explicit conversion from an arbitrary type to a svtkVariant.  Provides
 * callers with a "hook" for defining conversions from user-defined types to svtkVariant.
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkVariantCreate_h
#define svtkVariantCreate_h

#include <typeinfo> // for warnings

template <typename T>
svtkVariant svtkVariantCreate(const T&)
{
  svtkGenericWarningMacro(
    << "Cannot convert unsupported type [" << typeid(T).name() << "] to svtkVariant.  "
    << "Create a svtkVariantCreate<> specialization to eliminate this warning.");

  return svtkVariant();
}

template <>
inline svtkVariant svtkVariantCreate<char>(const char& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<unsigned char>(const unsigned char& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<short>(const short& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<unsigned short>(const unsigned short& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<int>(const int& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<unsigned int>(const unsigned int& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<long>(const long& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<unsigned long>(const unsigned long& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<long long>(const long long& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<unsigned long long>(const unsigned long long& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<float>(const float& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<double>(const double& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<svtkStdString>(const svtkStdString& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<svtkUnicodeString>(const svtkUnicodeString& value)
{
  return value;
}

template <>
inline svtkVariant svtkVariantCreate<svtkVariant>(const svtkVariant& value)
{
  return value;
}

#endif

// SVTK-HeaderTest-Exclude: svtkVariantCreate.h

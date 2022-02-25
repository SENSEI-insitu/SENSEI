/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArray.cxx

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

#include "svtkArray.h"
#include "svtkDenseArray.h"
#include "svtkSparseArray.h"
#include "svtkVariant.h"

#include <algorithm>

//
// Standard functions
//

//----------------------------------------------------------------------------

svtkArray::svtkArray() = default;

//----------------------------------------------------------------------------

svtkArray::~svtkArray() = default;

//----------------------------------------------------------------------------

void svtkArray::PrintSelf(ostream& os, svtkIndent indent)
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Name: " << this->Name << endl;

  os << indent << "Dimensions: " << this->GetDimensions() << endl;
  os << indent << "Extents: " << this->GetExtents() << endl;

  os << indent << "DimensionLabels:";
  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
    os << " " << this->GetDimensionLabel(i);
  os << endl;

  os << indent << "Size: " << this->GetSize() << endl;
  os << indent << "NonNullSize: " << this->GetNonNullSize() << endl;
}

svtkArray* svtkArray::CreateArray(int StorageType, int ValueType)
{
  switch (StorageType)
  {
    case DENSE:
    {
      switch (ValueType)
      {
        case SVTK_CHAR:
          return svtkDenseArray<char>::New();
        case SVTK_SIGNED_CHAR:
          return svtkDenseArray<signed char>::New();
        case SVTK_UNSIGNED_CHAR:
          return svtkDenseArray<unsigned char>::New();
        case SVTK_SHORT:
          return svtkDenseArray<short>::New();
        case SVTK_UNSIGNED_SHORT:
          return svtkDenseArray<unsigned short>::New();
        case SVTK_INT:
          return svtkDenseArray<int>::New();
        case SVTK_UNSIGNED_INT:
          return svtkDenseArray<unsigned int>::New();
        case SVTK_LONG:
          return svtkDenseArray<long>::New();
        case SVTK_UNSIGNED_LONG:
          return svtkDenseArray<unsigned long>::New();
        case SVTK_LONG_LONG:
          return svtkDenseArray<long long>::New();
        case SVTK_UNSIGNED_LONG_LONG:
          return svtkDenseArray<unsigned long long>::New();
        case SVTK_FLOAT:
          return svtkDenseArray<float>::New();
        case SVTK_DOUBLE:
          return svtkDenseArray<double>::New();
        case SVTK_ID_TYPE:
          return svtkDenseArray<svtkIdType>::New();
        case SVTK_STRING:
          return svtkDenseArray<svtkStdString>::New();
        case SVTK_UNICODE_STRING:
          return svtkDenseArray<svtkUnicodeString>::New();
        case SVTK_VARIANT:
          return svtkDenseArray<svtkVariant>::New();
      }
      svtkGenericWarningMacro(
        << "svtkArrary::CreateArray() cannot create array with unknown value type: "
        << svtkImageScalarTypeNameMacro(ValueType));
      return nullptr;
    }
    case SPARSE:
    {
      switch (ValueType)
      {
        case SVTK_CHAR:
          return svtkSparseArray<char>::New();
        case SVTK_SIGNED_CHAR:
          return svtkSparseArray<signed char>::New();
        case SVTK_UNSIGNED_CHAR:
          return svtkSparseArray<unsigned char>::New();
        case SVTK_SHORT:
          return svtkSparseArray<short>::New();
        case SVTK_UNSIGNED_SHORT:
          return svtkSparseArray<unsigned short>::New();
        case SVTK_INT:
          return svtkSparseArray<int>::New();
        case SVTK_UNSIGNED_INT:
          return svtkSparseArray<unsigned int>::New();
        case SVTK_LONG:
          return svtkSparseArray<long>::New();
        case SVTK_UNSIGNED_LONG:
          return svtkSparseArray<unsigned long>::New();
        case SVTK_LONG_LONG:
          return svtkSparseArray<long long>::New();
        case SVTK_UNSIGNED_LONG_LONG:
          return svtkSparseArray<unsigned long long>::New();
        case SVTK_FLOAT:
          return svtkSparseArray<float>::New();
        case SVTK_DOUBLE:
          return svtkSparseArray<double>::New();
        case SVTK_ID_TYPE:
          return svtkSparseArray<svtkIdType>::New();
        case SVTK_STRING:
          return svtkSparseArray<svtkStdString>::New();
        case SVTK_UNICODE_STRING:
          return svtkSparseArray<svtkUnicodeString>::New();
        case SVTK_VARIANT:
          return svtkSparseArray<svtkVariant>::New();
      }
      svtkGenericWarningMacro(
        << "svtkArrary::CreateArray() cannot create array with unknown value type: "
        << svtkImageScalarTypeNameMacro(ValueType));
      return nullptr;
    }
  }

  svtkGenericWarningMacro(
    << "svtkArrary::CreateArray() cannot create array with unknown storage type: " << StorageType);
  return nullptr;
}

void svtkArray::Resize(const CoordinateT i)
{
  this->Resize(svtkArrayExtents(svtkArrayRange(0, i)));
}

void svtkArray::Resize(const svtkArrayRange& i)
{
  this->Resize(svtkArrayExtents(i));
}

void svtkArray::Resize(const CoordinateT i, const CoordinateT j)
{
  this->Resize(svtkArrayExtents(svtkArrayRange(0, i), svtkArrayRange(0, j)));
}

void svtkArray::Resize(const svtkArrayRange& i, const svtkArrayRange& j)
{
  this->Resize(svtkArrayExtents(i, j));
}

void svtkArray::Resize(const CoordinateT i, const CoordinateT j, const CoordinateT k)
{
  this->Resize(svtkArrayExtents(svtkArrayRange(0, i), svtkArrayRange(0, j), svtkArrayRange(0, k)));
}

void svtkArray::Resize(const svtkArrayRange& i, const svtkArrayRange& j, const svtkArrayRange& k)
{
  this->Resize(svtkArrayExtents(i, j, k));
}

void svtkArray::Resize(const svtkArrayExtents& extents)
{
  this->InternalResize(extents);
}

svtkArrayRange svtkArray::GetExtent(DimensionT dimension)
{
  return this->GetExtents()[dimension];
}

svtkArray::DimensionT svtkArray::GetDimensions()
{
  return this->GetExtents().GetDimensions();
}

svtkTypeUInt64 svtkArray::GetSize()
{
  return this->GetExtents().GetSize();
}

void svtkArray::SetName(const svtkStdString& raw_name)
{
  // Don't allow newlines in array names ...
  svtkStdString name(raw_name);
  name.erase(std::remove(name.begin(), name.end(), '\r'), name.end());
  name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());

  this->Name = name;
}

svtkStdString svtkArray::GetName()
{
  return this->Name;
}

void svtkArray::SetDimensionLabel(DimensionT i, const svtkStdString& raw_label)
{
  if (i < 0 || i >= this->GetDimensions())
  {
    svtkErrorMacro(
      "Cannot set label for dimension " << i << " of a " << this->GetDimensions() << "-way array");
    return;
  }

  // Don't allow newlines in dimension labels ...
  svtkStdString label(raw_label);
  label.erase(std::remove(label.begin(), label.end(), '\r'), label.end());
  label.erase(std::remove(label.begin(), label.end(), '\n'), label.end());

  this->InternalSetDimensionLabel(i, label);
}

svtkStdString svtkArray::GetDimensionLabel(DimensionT i)
{
  if (i < 0 || i >= this->GetDimensions())
  {
    svtkErrorMacro(
      "Cannot get label for dimension " << i << " of a " << this->GetDimensions() << "-way array");
    return "";
  }

  return this->InternalGetDimensionLabel(i);
}

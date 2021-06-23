/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDenseArray.txx

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

#ifndef svtkDenseArray_txx
#define svtkDenseArray_txx

#include "svtkObjectFactory.h"

///////////////////////////////////////////////////////////////////////////////
// svtkDenseArray::MemoryBlock

template <typename T>
svtkDenseArray<T>::MemoryBlock::~MemoryBlock()
{
}

///////////////////////////////////////////////////////////////////////////////
// svtkDenseArray::HeapMemoryBlock

template <typename T>
svtkDenseArray<T>::HeapMemoryBlock::HeapMemoryBlock(const svtkArrayExtents& extents)
  : Storage(new T[extents.GetSize()])
{
}

template <typename T>
svtkDenseArray<T>::HeapMemoryBlock::~HeapMemoryBlock()
{
  delete[] this->Storage;
}

template <typename T>
T* svtkDenseArray<T>::HeapMemoryBlock::GetAddress()
{
  return this->Storage;
}

///////////////////////////////////////////////////////////////////////////////
// svtkDenseArray::StaticMemoryBlock

template <typename T>
svtkDenseArray<T>::StaticMemoryBlock::StaticMemoryBlock(T* const storage)
  : Storage(storage)
{
}

template <typename T>
T* svtkDenseArray<T>::StaticMemoryBlock::GetAddress()
{
  return this->Storage;
}

///////////////////////////////////////////////////////////////////////////////
// svtkDenseArray

template <typename T>
svtkDenseArray<T>* svtkDenseArray<T>::New()
{
  // Don't use object factory macros on templated classes. It'll confuse the
  // object factory.
  svtkDenseArray<T>* ret = new svtkDenseArray<T>;
  ret->InitializeObjectBase();
  return ret;
}

template <typename T>
void svtkDenseArray<T>::PrintSelf(ostream& os, svtkIndent indent)
{
  svtkDenseArray<T>::Superclass::PrintSelf(os, indent);
}

template <typename T>
bool svtkDenseArray<T>::IsDense()
{
  return true;
}

template <typename T>
const svtkArrayExtents& svtkDenseArray<T>::GetExtents()
{
  return this->Extents;
}

template <typename T>
typename svtkDenseArray<T>::SizeT svtkDenseArray<T>::GetNonNullSize()
{
  return this->Extents.GetSize();
}

template <typename T>
void svtkDenseArray<T>::GetCoordinatesN(const SizeT n, svtkArrayCoordinates& coordinates)
{
  coordinates.SetDimensions(this->GetDimensions());

  svtkIdType divisor = 1;
  for (DimensionT i = 0; i < this->GetDimensions(); ++i)
  {
    coordinates[i] = ((n / divisor) % this->Extents[i].GetSize()) + this->Extents[i].GetBegin();
    divisor *= this->Extents[i].GetSize();
  }
}

template <typename T>
svtkArray* svtkDenseArray<T>::DeepCopy()
{
  svtkDenseArray<T>* const copy = svtkDenseArray<T>::New();

  copy->SetName(this->GetName());
  copy->Resize(this->Extents);
  copy->DimensionLabels = this->DimensionLabels;
  std::copy(this->Begin, this->End, copy->Begin);

  return copy;
}

template <typename T>
const T& svtkDenseArray<T>::GetValue(CoordinateT i)
{
  if (1 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    static T temp;
    return temp;
  }

  return this->Begin[this->MapCoordinates(i)];
}

template <typename T>
const T& svtkDenseArray<T>::GetValue(CoordinateT i, CoordinateT j)
{
  if (2 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    static T temp;
    return temp;
  }

  return this->Begin[this->MapCoordinates(i, j)];
}

template <typename T>
const T& svtkDenseArray<T>::GetValue(CoordinateT i, CoordinateT j, CoordinateT k)
{
  if (3 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    static T temp;
    return temp;
  }

  return this->Begin[this->MapCoordinates(i, j, k)];
}

template <typename T>
const T& svtkDenseArray<T>::GetValue(const svtkArrayCoordinates& coordinates)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    static T temp;
    return temp;
  }

  return this->Begin[this->MapCoordinates(coordinates)];
}

template <typename T>
const T& svtkDenseArray<T>::GetValueN(const SizeT n)
{
  return this->Begin[n];
}

template <typename T>
void svtkDenseArray<T>::SetValue(CoordinateT i, const T& value)
{
  if (1 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  this->Begin[this->MapCoordinates(i)] = value;
}

template <typename T>
void svtkDenseArray<T>::SetValue(CoordinateT i, CoordinateT j, const T& value)
{
  if (2 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  this->Begin[this->MapCoordinates(i, j)] = value;
}

template <typename T>
void svtkDenseArray<T>::SetValue(CoordinateT i, CoordinateT j, CoordinateT k, const T& value)
{
  if (3 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  this->Begin[this->MapCoordinates(i, j, k)] = value;
}

template <typename T>
void svtkDenseArray<T>::SetValue(const svtkArrayCoordinates& coordinates, const T& value)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  this->Begin[this->MapCoordinates(coordinates)] = value;
}

template <typename T>
void svtkDenseArray<T>::SetValueN(const SizeT n, const T& value)
{
  this->Begin[n] = value;
}

template <typename T>
void svtkDenseArray<T>::ExternalStorage(const svtkArrayExtents& extents, MemoryBlock* storage)
{
  this->Reconfigure(extents, storage);
}

template <typename T>
void svtkDenseArray<T>::Fill(const T& value)
{
  std::fill(this->Begin, this->End, value);
}

template <typename T>
T& svtkDenseArray<T>::operator[](const svtkArrayCoordinates& coordinates)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    static T temp;
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return temp;
  }

  return this->Begin[this->MapCoordinates(coordinates)];
}

template <typename T>
const T* svtkDenseArray<T>::GetStorage() const
{
  return this->Begin;
}

template <typename T>
T* svtkDenseArray<T>::GetStorage()
{
  return this->Begin;
}

template <typename T>
svtkDenseArray<T>::svtkDenseArray()
  : Storage(nullptr)
  , Begin(nullptr)
  , End(nullptr)
{
}

template <typename T>
svtkDenseArray<T>::~svtkDenseArray()
{
  delete this->Storage;

  this->Storage = nullptr;
  this->Begin = nullptr;
  this->End = nullptr;
}

template <typename T>
void svtkDenseArray<T>::InternalResize(const svtkArrayExtents& extents)
{
  this->Reconfigure(extents, new HeapMemoryBlock(extents));
}

template <typename T>
void svtkDenseArray<T>::InternalSetDimensionLabel(DimensionT i, const svtkStdString& label)
{
  this->DimensionLabels[i] = label;
}

template <typename T>
svtkStdString svtkDenseArray<T>::InternalGetDimensionLabel(DimensionT i)
{
  return this->DimensionLabels[i];
}

template <typename T>
svtkIdType svtkDenseArray<T>::MapCoordinates(CoordinateT i)
{
  return ((i + this->Offsets[0]) * this->Strides[0]);
}

template <typename T>
svtkIdType svtkDenseArray<T>::MapCoordinates(CoordinateT i, CoordinateT j)
{
  return ((i + this->Offsets[0]) * this->Strides[0]) + ((j + this->Offsets[1]) * this->Strides[1]);
}

template <typename T>
svtkIdType svtkDenseArray<T>::MapCoordinates(CoordinateT i, CoordinateT j, CoordinateT k)
{
  return ((i + this->Offsets[0]) * this->Strides[0]) + ((j + this->Offsets[1]) * this->Strides[1]) +
    ((k + this->Offsets[2]) * this->Strides[2]);
}

template <typename T>
svtkIdType svtkDenseArray<T>::MapCoordinates(const svtkArrayCoordinates& coordinates)
{
  svtkIdType index = 0;
  for (svtkIdType i = 0; i != static_cast<svtkIdType>(this->Strides.size()); ++i)
    index += ((coordinates[i] + this->Offsets[i]) * this->Strides[i]);

  return index;
}

template <typename T>
void svtkDenseArray<T>::Reconfigure(const svtkArrayExtents& extents, MemoryBlock* storage)
{
  this->Extents = extents;
  this->DimensionLabels.resize(extents.GetDimensions(), svtkStdString());

  delete this->Storage;
  this->Storage = storage;
  this->Begin = storage->GetAddress();
  this->End = this->Begin + extents.GetSize();

  this->Offsets.resize(extents.GetDimensions());
  for (DimensionT i = 0; i != extents.GetDimensions(); ++i)
  {
    this->Offsets[i] = -extents[i].GetBegin();
  }

  this->Strides.resize(extents.GetDimensions());
  for (DimensionT i = 0; i != extents.GetDimensions(); ++i)
  {
    if (i == 0)
      this->Strides[i] = 1;
    else
      this->Strides[i] = this->Strides[i - 1] * extents[i - 1].GetSize();
  }
}

#endif

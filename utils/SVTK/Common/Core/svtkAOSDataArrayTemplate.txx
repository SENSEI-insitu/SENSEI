/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAOSDataArrayTemplate.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkAOSDataArrayTemplate_txx
#define svtkAOSDataArrayTemplate_txx

#include "svtkAOSDataArrayTemplate.h"

#include "svtkArrayIteratorTemplate.h"

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkAOSDataArrayTemplate<ValueTypeT>* svtkAOSDataArrayTemplate<ValueTypeT>::New()
{
  SVTK_STANDARD_NEW_BODY(svtkAOSDataArrayTemplate<ValueType>);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkAOSDataArrayTemplate<ValueTypeT>::svtkAOSDataArrayTemplate()
{
  this->Buffer = svtkBuffer<ValueType>::New();
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkAOSDataArrayTemplate<ValueTypeT>::~svtkAOSDataArrayTemplate()
{
  this->Buffer->Delete();
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetArray(
  ValueType* array, svtkIdType size, int save, int deleteMethod)
{

  this->Buffer->SetBuffer(array, size);

  if (deleteMethod == SVTK_DATA_ARRAY_DELETE)
  {
    this->Buffer->SetFreeFunction(save != 0, ::operator delete[]);
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_ALIGNED_FREE)
  {
#ifdef _WIN32
    this->Buffer->SetFreeFunction(save != 0, _aligned_free);
#else
    this->Buffer->SetFreeFunction(save != 0, free);
#endif
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_USER_DEFINED || deleteMethod == SVTK_DATA_ARRAY_FREE)
  {
    this->Buffer->SetFreeFunction(save != 0, free);
  }

  this->Size = size;
  this->MaxId = this->Size - 1;
  this->DataChanged();
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetArray(ValueType* array, svtkIdType size, int save)
{
  this->SetArray(array, size, save, SVTK_DATA_ARRAY_FREE);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetVoidArray(void* array, svtkIdType size, int save)
{
  this->SetArray(static_cast<ValueType*>(array), size, save);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetVoidArray(
  void* array, svtkIdType size, int save, int deleteMethod)
{
  this->SetArray(static_cast<ValueType*>(array), size, save, deleteMethod);
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkAOSDataArrayTemplate<ValueType>::SetArrayFreeFunction(void (*callback)(void*))
{
  this->Buffer->SetFreeFunction(false, callback);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetTuple(svtkIdType tupleIdx, const float* tuple)
{
  // While std::copy is the obvious choice here, it kills performance on MSVC
  // debugging builds as their STL calls are poorly optimized. Just use a for
  // loop instead.
  ValueTypeT* data = this->Buffer->GetBuffer() + tupleIdx * this->NumberOfComponents;
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    data[i] = static_cast<ValueType>(tuple[i]);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::SetTuple(svtkIdType tupleIdx, const double* tuple)
{
  // See note in SetTuple about std::copy vs for loops on MSVC.
  ValueTypeT* data = this->Buffer->GetBuffer() + tupleIdx * this->NumberOfComponents;
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    data[i] = static_cast<ValueType>(tuple[i]);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::InsertTuple(svtkIdType tupleIdx, const float* tuple)
{
  if (this->EnsureAccessToTuple(tupleIdx))
  {
    // See note in SetTuple about std::copy vs for loops on MSVC.
    const svtkIdType valueIdx = tupleIdx * this->NumberOfComponents;
    ValueTypeT* data = this->Buffer->GetBuffer() + valueIdx;
    for (int i = 0; i < this->NumberOfComponents; ++i)
    {
      data[i] = static_cast<ValueType>(tuple[i]);
    }
    this->MaxId = std::max(this->MaxId, valueIdx + this->NumberOfComponents - 1);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::InsertTuple(svtkIdType tupleIdx, const double* tuple)
{
  if (this->EnsureAccessToTuple(tupleIdx))
  {
    // See note in SetTuple about std::copy vs for loops on MSVC.
    const svtkIdType valueIdx = tupleIdx * this->NumberOfComponents;
    ValueTypeT* data = this->Buffer->GetBuffer() + valueIdx;
    for (int i = 0; i < this->NumberOfComponents; ++i)
    {
      data[i] = static_cast<ValueType>(tuple[i]);
    }
    this->MaxId = std::max(this->MaxId, valueIdx + this->NumberOfComponents - 1);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::InsertComponent(
  svtkIdType tupleIdx, int compIdx, double value)
{
  const svtkIdType newMaxId = tupleIdx * this->NumberOfComponents + compIdx;
  if (newMaxId >= this->Size)
  {
    if (!this->Resize(newMaxId / this->NumberOfComponents + 1))
    {
      return;
    }
  }

  this->Buffer->GetBuffer()[newMaxId] = static_cast<ValueTypeT>(value);
  this->MaxId = std::max(newMaxId, this->MaxId);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkIdType svtkAOSDataArrayTemplate<ValueTypeT>::InsertNextTuple(const float* tuple)
{
  svtkIdType newMaxId = this->MaxId + this->NumberOfComponents;
  const svtkIdType tupleIdx = newMaxId / this->NumberOfComponents;
  if (newMaxId >= this->Size)
  {
    if (!this->Resize(tupleIdx + 1))
    {
      return -1;
    }
  }

  // See note in SetTuple about std::copy vs for loops on MSVC.
  ValueTypeT* data = this->Buffer->GetBuffer() + this->MaxId + 1;
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    data[i] = static_cast<ValueType>(tuple[i]);
  }
  this->MaxId = newMaxId;
  return tupleIdx;
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkIdType svtkAOSDataArrayTemplate<ValueTypeT>::InsertNextTuple(const double* tuple)
{
  svtkIdType newMaxId = this->MaxId + this->NumberOfComponents;
  const svtkIdType tupleIdx = newMaxId / this->NumberOfComponents;
  if (newMaxId >= this->Size)
  {
    if (!this->Resize(tupleIdx + 1))
    {
      return -1;
    }
  }

  // See note in SetTuple about std::copy vs for loops on MSVC.
  ValueTypeT* data = this->Buffer->GetBuffer() + this->MaxId + 1;
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    data[i] = static_cast<ValueType>(tuple[i]);
  }
  this->MaxId = newMaxId;
  return tupleIdx;
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::GetTuple(svtkIdType tupleIdx, double* tuple)
{
  ValueTypeT* data = this->Buffer->GetBuffer() + tupleIdx * this->NumberOfComponents;
  // See note in SetTuple about std::copy vs for loops on MSVC.
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    tuple[i] = static_cast<double>(data[i]);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
double* svtkAOSDataArrayTemplate<ValueTypeT>::GetTuple(svtkIdType tupleIdx)
{
  ValueTypeT* data = this->Buffer->GetBuffer() + tupleIdx * this->NumberOfComponents;
  double* tuple = &this->LegacyTuple[0];
  // See note in SetTuple about std::copy vs for loops on MSVC.
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    tuple[i] = static_cast<double>(data[i]);
  }
  return &this->LegacyTuple[0];
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
svtkArrayIterator* svtkAOSDataArrayTemplate<ValueTypeT>::NewIterator()
{
  svtkArrayIterator* iter = svtkArrayIteratorTemplate<ValueType>::New();
  iter->Initialize(this);
  return iter;
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::ShallowCopy(svtkDataArray* other)
{
  SelfType* o = SelfType::FastDownCast(other);
  if (o)
  {
    this->Size = o->Size;
    this->MaxId = o->MaxId;
    this->SetName(o->Name);
    this->SetNumberOfComponents(o->NumberOfComponents);
    this->CopyComponentNames(o);
    if (this->Buffer != o->Buffer)
    {
      this->Buffer->Delete();
      this->Buffer = o->Buffer;
      this->Buffer->Register(nullptr);
    }
    this->DataChanged();
  }
  else
  {
    this->Superclass::ShallowCopy(other);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  SelfType* other = svtkArrayDownCast<SelfType>(source);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::InsertTuples(dstStart, n, srcStart, source);
    return;
  }

  if (n == 0)
  {
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << other->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  svtkIdType maxSrcTupleId = srcStart + n - 1;
  svtkIdType maxDstTupleId = dstStart + n - 1;

  if (maxSrcTupleId >= other->GetNumberOfTuples())
  {
    svtkErrorMacro("Source array too small, requested tuple at index "
      << maxSrcTupleId << ", but there are only " << other->GetNumberOfTuples()
      << " tuples in the array.");
    return;
  }

  svtkIdType newSize = (maxDstTupleId + 1) * this->NumberOfComponents;
  if (this->Size < newSize)
  {
    if (!this->Resize(maxDstTupleId + 1))
    {
      svtkErrorMacro("Resize failed.");
      return;
    }
  }

  this->MaxId = std::max(this->MaxId, newSize - 1);

  ValueType* srcBegin = other->GetPointer(srcStart * numComps);
  ValueType* srcEnd = srcBegin + (n * numComps);
  ValueType* dstBegin = this->GetPointer(dstStart * numComps);

  std::copy(srcBegin, srcEnd, dstBegin);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::FillTypedComponent(int compIdx, ValueType value)
{
  if (this->NumberOfComponents <= 1)
  {
    this->FillValue(value);
  }
  else
  {
    this->Superclass::FillTypedComponent(compIdx, value);
  }
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::FillValue(ValueType value)
{
  std::ptrdiff_t offset = this->MaxId + 1;
  std::fill(this->Buffer->GetBuffer(), this->Buffer->GetBuffer() + offset, value);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void svtkAOSDataArrayTemplate<ValueTypeT>::Fill(double value)
{
  this->FillValue(static_cast<ValueType>(value));
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
typename svtkAOSDataArrayTemplate<ValueTypeT>::ValueType*
svtkAOSDataArrayTemplate<ValueTypeT>::WritePointer(svtkIdType valueIdx, svtkIdType numValues)
{
  svtkIdType newSize = valueIdx + numValues;
  if (newSize > this->Size)
  {
    if (!this->Resize(newSize / this->NumberOfComponents + 1))
    {
      return nullptr;
    }
    this->MaxId = (newSize - 1);
  }

  // For extending the in-use ids but not the size:
  this->MaxId = std::max(this->MaxId, newSize - 1);

  this->DataChanged();
  return this->GetPointer(valueIdx);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void* svtkAOSDataArrayTemplate<ValueTypeT>::WriteVoidPointer(svtkIdType valueIdx, svtkIdType numValues)
{
  return this->WritePointer(valueIdx, numValues);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
typename svtkAOSDataArrayTemplate<ValueTypeT>::ValueType*
svtkAOSDataArrayTemplate<ValueTypeT>::GetPointer(svtkIdType valueIdx)
{
  return this->Buffer->GetBuffer() + valueIdx;
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
void* svtkAOSDataArrayTemplate<ValueTypeT>::GetVoidPointer(svtkIdType valueIdx)
{
  return this->GetPointer(valueIdx);
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
bool svtkAOSDataArrayTemplate<ValueTypeT>::AllocateTuples(svtkIdType numTuples)
{
  svtkIdType numValues = numTuples * this->GetNumberOfComponents();
  if (this->Buffer->Allocate(numValues))
  {
    this->Size = this->Buffer->GetSize();
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
template <class ValueTypeT>
bool svtkAOSDataArrayTemplate<ValueTypeT>::ReallocateTuples(svtkIdType numTuples)
{
  if (this->Buffer->Reallocate(numTuples * this->GetNumberOfComponents()))
  {
    this->Size = this->Buffer->GetSize();
    return true;
  }
  return false;
}

#endif // header guard

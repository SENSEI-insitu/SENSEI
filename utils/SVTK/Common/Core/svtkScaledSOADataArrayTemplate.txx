/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkScaledSOADataArrayTemplate.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkScaledSOADataArrayTemplate_txx
#define svtkScaledSOADataArrayTemplate_txx

#include "svtkScaledSOADataArrayTemplate.h"

#include "svtkArrayIteratorTemplate.h"
#include "svtkBuffer.h"

#include <cassert>

//-----------------------------------------------------------------------------
template <class ValueType>
svtkScaledSOADataArrayTemplate<ValueType>* svtkScaledSOADataArrayTemplate<ValueType>::New()
{
  SVTK_STANDARD_NEW_BODY(svtkScaledSOADataArrayTemplate<ValueType>);
}

//-----------------------------------------------------------------------------
template <class ValueType>
svtkScaledSOADataArrayTemplate<ValueType>::svtkScaledSOADataArrayTemplate()
  : AoSCopy(nullptr)
  , Scale(1)
{
}

//-----------------------------------------------------------------------------
template <class ValueType>
svtkScaledSOADataArrayTemplate<ValueType>::~svtkScaledSOADataArrayTemplate()
{
  for (size_t cc = 0; cc < this->Data.size(); ++cc)
  {
    this->Data[cc]->Delete();
  }
  this->Data.clear();
  if (this->AoSCopy)
  {
    this->AoSCopy->Delete();
    this->AoSCopy = nullptr;
  }
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::SetNumberOfComponents(int val)
{
  this->GenericDataArrayType::SetNumberOfComponents(val);
  size_t numComps = static_cast<size_t>(this->GetNumberOfComponents());
  assert(numComps >= 1);
  while (this->Data.size() > numComps)
  {
    this->Data.back()->Delete();
    this->Data.pop_back();
  }
  while (this->Data.size() < numComps)
  {
    this->Data.push_back(svtkBuffer<ValueType>::New());
  }
}

//-----------------------------------------------------------------------------
template <class ValueType>
svtkArrayIterator* svtkScaledSOADataArrayTemplate<ValueType>::NewIterator()
{
  svtkArrayIterator* iter = svtkArrayIteratorTemplate<ValueType>::New();
  iter->Initialize(this);
  return iter;
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::ShallowCopy(svtkDataArray* other)
{
  SelfType* o = SelfType::FastDownCast(other);
  if (o)
  {
    this->Size = o->Size;
    this->MaxId = o->MaxId;
    this->SetName(o->Name);
    this->SetNumberOfComponents(o->NumberOfComponents);
    this->CopyComponentNames(o);
    this->Scale = o->Scale;
    assert(this->Data.size() == o->Data.size());
    for (size_t cc = 0; cc < this->Data.size(); ++cc)
    {
      svtkBuffer<ValueType>* thisBuffer = this->Data[cc];
      svtkBuffer<ValueType>* otherBuffer = o->Data[cc];
      if (thisBuffer != otherBuffer)
      {
        thisBuffer->Delete();
        this->Data[cc] = otherBuffer;
        otherBuffer->Register(nullptr);
      }
    }
    this->DataChanged();
  }
  else
  {
    this->Superclass::ShallowCopy(other);
  }
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::InsertTuples(
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

  std::vector<ValueType> vals(numComps);
  for (svtkIdType i = 0; i < n; i++)
  {
    other->GetTypedTuple(i + srcStart, vals.data());
    this->SetTypedTuple(i + dstStart, vals.data()); // will automatically scale data
  }
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::FillTypedComponent(int compIdx, ValueType value)
{
  ValueType* buffer = this->Data[compIdx]->GetBuffer();
  value /= this->Scale;
  std::fill(buffer, buffer + this->GetNumberOfTuples(), value);
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::SetArray(
  int comp, ValueType* array, svtkIdType size, bool updateMaxId, bool save, int deleteMethod)
{
  const int numComps = this->GetNumberOfComponents();
  if (comp >= numComps || comp < 0)
  {
    svtkErrorMacro("Invalid component number '"
      << comp
      << "' specified. "
         "Use `SetNumberOfComponents` first to set the number of components.");
    return;
  }

  this->Data[comp]->SetBuffer(array, size);

  if (deleteMethod == SVTK_DATA_ARRAY_DELETE)
  {
    this->Data[comp]->SetFreeFunction(save != 0, ::operator delete[]);
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_ALIGNED_FREE)
  {
#ifdef _WIN32
    this->Data[comp]->SetFreeFunction(save != 0, _aligned_free);
#else
    this->Data[comp]->SetFreeFunction(save != 0, free);
#endif
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_USER_DEFINED || deleteMethod == SVTK_DATA_ARRAY_FREE)
  {
    this->Data[comp]->SetFreeFunction(save != 0, free);
  }

  if (updateMaxId)
  {
    this->Size = numComps * size;
    this->MaxId = this->Size - 1;
  }
  this->DataChanged();
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::SetArrayFreeFunction(void (*callback)(void*))
{
  const int numComps = this->GetNumberOfComponents();
  for (int i = 0; i < numComps; ++i)
  {
    this->SetArrayFreeFunction(i, callback);
  }
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::SetArrayFreeFunction(
  int comp, void (*callback)(void*))
{
  const int numComps = this->GetNumberOfComponents();
  if (comp >= numComps || comp < 0)
  {
    svtkErrorMacro("Invalid component number '"
      << comp
      << "' specified. "
         "Use `SetNumberOfComponents` first to set the number of components.");
    return;
  }
  this->Data[comp]->SetFreeFunction(false, callback);
}

//-----------------------------------------------------------------------------
template <class ValueType>
typename svtkScaledSOADataArrayTemplate<ValueType>::ValueType*
svtkScaledSOADataArrayTemplate<ValueType>::GetComponentArrayPointer(int comp)
{
  const int numComps = this->GetNumberOfComponents();
  if (comp >= numComps || comp < 0)
  {
    svtkErrorMacro("Invalid component number '" << comp << "' specified.");
    return nullptr;
  }

  return this->Data[comp]->GetBuffer();
}

//-----------------------------------------------------------------------------
template <class ValueType>
bool svtkScaledSOADataArrayTemplate<ValueType>::AllocateTuples(svtkIdType numTuples)
{
  for (size_t cc = 0, max = this->Data.size(); cc < max; ++cc)
  {
    if (!this->Data[cc]->Allocate(numTuples))
    {
      return false;
    }
  }
  return true;
}

//-----------------------------------------------------------------------------
template <class ValueType>
bool svtkScaledSOADataArrayTemplate<ValueType>::ReallocateTuples(svtkIdType numTuples)
{
  for (size_t cc = 0, max = this->Data.size(); cc < max; ++cc)
  {
    if (!this->Data[cc]->Reallocate(numTuples))
    {
      return false;
    }
  }
  return true;
}

//-----------------------------------------------------------------------------
template <class ValueType>
void* svtkScaledSOADataArrayTemplate<ValueType>::GetVoidPointer(svtkIdType valueIdx)
{
  // Allow warnings to be silenced:
  const char* silence = getenv("SVTK_SILENCE_GET_VOID_POINTER_WARNINGS");
  if (!silence)
  {
    svtkWarningMacro(<< "GetVoidPointer called. This is very expensive for "
                       "non-array-of-structs subclasses, as the scalar array "
                       "must be generated for each call. Using the "
                       "svtkGenericDataArray API with svtkArrayDispatch are "
                       "preferred. Define the environment variable "
                       "SVTK_SILENCE_GET_VOID_POINTER_WARNINGS to silence "
                       "this warning. Additionally, for the svtkScaledSOADataArrayTemplate "
                       "class we also set Scale to 1 since we've scaled how "
                       "we're storing the data in memory now. ");
  }

  size_t numValues = this->GetNumberOfValues();

  if (!this->AoSCopy)
  {
    this->AoSCopy = svtkBuffer<ValueType>::New();
  }

  if (!this->AoSCopy->Allocate(static_cast<svtkIdType>(numValues)))
  {
    svtkErrorMacro(<< "Error allocating a buffer of " << numValues << " '"
                  << this->GetDataTypeAsString() << "' elements.");
    return nullptr;
  }

  this->ExportToVoidPointer(static_cast<void*>(this->AoSCopy->GetBuffer()));

  // This is the hacky thing with this class that we now need to set the scale
  // to 1 since we internally are storing the memory in an unscaled manner
  this->Scale = 1.0;

  return static_cast<void*>(this->AoSCopy->GetBuffer() + valueIdx);
}

//-----------------------------------------------------------------------------
template <class ValueType>
void svtkScaledSOADataArrayTemplate<ValueType>::ExportToVoidPointer(void* voidPtr)
{
  svtkIdType numTuples = this->GetNumberOfTuples();
  if (this->NumberOfComponents * numTuples == 0)
  {
    // Nothing to do.
    return;
  }

  if (!voidPtr)
  {
    svtkErrorMacro(<< "Buffer is nullptr.");
    return;
  }

  ValueType* ptr = static_cast<ValueType*>(voidPtr);
  for (svtkIdType t = 0; t < numTuples; ++t)
  {
    for (int c = 0; c < this->NumberOfComponents; ++c)
    {
      *ptr++ = this->Data[c]->GetBuffer()[t] * this->Scale;
    }
  }
}

#endif

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericDataArray.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkGenericDataArray_txx
#define svtkGenericDataArray_txx

#include "svtkGenericDataArray.h"

#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkVariantCast.h"

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
double* svtkGenericDataArray<DerivedT, ValueTypeT>::GetTuple(svtkIdType tupleIdx)
{
  assert(!this->LegacyTuple.empty() && "Number of components is nonzero.");
  this->GetTuple(tupleIdx, &this->LegacyTuple[0]);
  return &this->LegacyTuple[0];
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::GetTuple(svtkIdType tupleIdx, double* tuple)
{
  for (int c = 0; c < this->NumberOfComponents; ++c)
  {
    tuple[c] = static_cast<double>(this->GetTypedComponent(tupleIdx, c));
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InterpolateTuple(
  svtkIdType dstTupleIdx, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other = svtkArrayDownCast<DerivedT>(source);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::InterpolateTuple(dstTupleIdx, ptIndices, source, weights);
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << other->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  svtkIdType numIds = ptIndices->GetNumberOfIds();
  svtkIdType* ids = ptIndices->GetPointer(0);

  for (int c = 0; c < numComps; ++c)
  {
    double val = 0.;
    for (svtkIdType tupleId = 0; tupleId < numIds; ++tupleId)
    {
      svtkIdType t = ids[tupleId];
      double weight = weights[tupleId];
      val += weight * static_cast<double>(other->GetTypedComponent(t, c));
    }
    ValueType valT;
    svtkMath::RoundDoubleToIntegralIfNecessary(val, &valT);
    this->InsertTypedComponent(dstTupleIdx, c, valT);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InterpolateTuple(svtkIdType dstTupleIdx,
  svtkIdType srcTupleIdx1, svtkAbstractArray* source1, svtkIdType srcTupleIdx2,
  svtkAbstractArray* source2, double t)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other1 = svtkArrayDownCast<DerivedT>(source1);
  DerivedT* other2 = other1 ? svtkArrayDownCast<DerivedT>(source2) : nullptr;
  if (!other1 || !other2)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::InterpolateTuple(
      dstTupleIdx, srcTupleIdx1, source1, srcTupleIdx2, source2, t);
    return;
  }

  if (srcTupleIdx1 >= source1->GetNumberOfTuples())
  {
    svtkErrorMacro("Tuple 1 out of range for provided array. "
                  "Requested tuple: "
      << srcTupleIdx1
      << " "
         "Tuples: "
      << source1->GetNumberOfTuples());
    return;
  }

  if (srcTupleIdx2 >= source2->GetNumberOfTuples())
  {
    svtkErrorMacro("Tuple 2 out of range for provided array. "
                  "Requested tuple: "
      << srcTupleIdx2
      << " "
         "Tuples: "
      << source2->GetNumberOfTuples());
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other1->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << other1->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }
  if (other2->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << other2->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  const double oneMinusT = 1. - t;
  double val;
  ValueType valT;

  for (int c = 0; c < numComps; ++c)
  {
    val = other1->GetTypedComponent(srcTupleIdx1, c) * oneMinusT +
      other2->GetTypedComponent(srcTupleIdx2, c) * t;
    svtkMath::RoundDoubleToIntegralIfNecessary(val, &valT);
    this->InsertTypedComponent(dstTupleIdx, c, valT);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetComponent(
  svtkIdType tupleIdx, int compIdx, double value)
{
  // Reimplemented for efficiency (base impl allocates heap memory)
  this->SetTypedComponent(tupleIdx, compIdx, static_cast<ValueType>(value));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
double svtkGenericDataArray<DerivedT, ValueTypeT>::GetComponent(svtkIdType tupleIdx, int compIdx)
{
  // Reimplemented for efficiency (base impl allocates heap memory)
  return static_cast<double>(this->GetTypedComponent(tupleIdx, compIdx));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::RemoveTuple(svtkIdType id)
{
  if (id < 0 || id >= this->GetNumberOfTuples())
  {
    // Nothing to be done
    return;
  }
  if (id == (this->GetNumberOfTuples() - 1))
  {
    // To remove last item, just decrease the size by one
    this->RemoveLastTuple();
    return;
  }

  // This is a very slow implementation since it uses generic API. Subclasses
  // are encouraged to provide a faster implementation.
  assert(((this->GetNumberOfTuples() - id) - 1) /* (length) */ > 0);

  int numComps = this->GetNumberOfComponents();
  svtkIdType fromTuple = id + 1;
  svtkIdType toTuple = id;
  svtkIdType endTuple = this->GetNumberOfTuples();
  for (; fromTuple != endTuple; ++toTuple, ++fromTuple)
  {
    for (int comp = 0; comp < numComps; ++comp)
    {
      this->SetTypedComponent(toTuple, comp, this->GetTypedComponent(fromTuple, comp));
    }
  }
  this->SetNumberOfTuples(this->GetNumberOfTuples() - 1);
  this->DataChanged();
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetVoidArray(void*, svtkIdType, int)
{
  svtkErrorMacro("SetVoidArray is not supported by this class.");
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetVoidArray(void*, svtkIdType, int, int)
{
  svtkErrorMacro("SetVoidArray is not supported by this class.");
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetArrayFreeFunction(void (*)(void*))
{
  svtkErrorMacro("SetArrayFreeFunction is not supported by this class.");
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void* svtkGenericDataArray<DerivedT, ValueTypeT>::WriteVoidPointer(svtkIdType, svtkIdType)
{
  svtkErrorMacro("WriteVoidPointer is not supported by this class.");
  return nullptr;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
typename svtkGenericDataArray<DerivedT, ValueTypeT>::ValueType*
svtkGenericDataArray<DerivedT, ValueTypeT>::WritePointer(svtkIdType id, svtkIdType number)
{
  return static_cast<ValueType*>(this->WriteVoidPointer(id, number));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
int svtkGenericDataArray<DerivedT, ValueTypeT>::GetDataType() const
{
  return svtkTypeTraits<ValueType>::SVTK_TYPE_ID;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
int svtkGenericDataArray<DerivedT, ValueTypeT>::GetDataTypeSize() const
{
  return static_cast<int>(sizeof(ValueType));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::HasStandardMemoryLayout() const
{
  // False by default, AoS should set true.
  return false;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void* svtkGenericDataArray<DerivedT, ValueTypeT>::GetVoidPointer(svtkIdType)
{
  svtkErrorMacro("GetVoidPointer is not supported by this class.");
  return nullptr;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
typename svtkGenericDataArray<DerivedT, ValueTypeT>::ValueType*
svtkGenericDataArray<DerivedT, ValueTypeT>::GetPointer(svtkIdType id)
{
  return static_cast<ValueType*>(this->GetVoidPointer(id));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::LookupValue(svtkVariant valueVariant)
{
  bool valid = true;
  ValueType value = svtkVariantCast<ValueType>(valueVariant, &valid);
  if (valid)
  {
    return this->LookupTypedValue(value);
  }
  return -1;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::LookupTypedValue(ValueType value)
{
  return this->Lookup.LookupValue(value);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::LookupValue(svtkVariant valueVariant, svtkIdList* ids)
{
  ids->Reset();
  bool valid = true;
  ValueType value = svtkVariantCast<ValueType>(valueVariant, &valid);
  if (valid)
  {
    this->LookupTypedValue(value, ids);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::LookupTypedValue(ValueType value, svtkIdList* ids)
{
  ids->Reset();
  this->Lookup.LookupValue(value, ids);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::ClearLookup()
{
  this->Lookup.ClearLookup();
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::DataChanged()
{
  this->Lookup.ClearLookup();
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetVariantValue(
  svtkIdType valueIdx, svtkVariant valueVariant)
{
  bool valid = true;
  ValueType value = svtkVariantCast<ValueType>(valueVariant, &valid);
  if (valid)
  {
    this->SetValue(valueIdx, value);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkVariant svtkGenericDataArray<DerivedT, ValueTypeT>::GetVariantValue(svtkIdType valueIdx)
{
  return svtkVariant(this->GetValue(valueIdx));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertVariantValue(
  svtkIdType valueIdx, svtkVariant valueVariant)
{
  bool valid = true;
  ValueType value = svtkVariantCast<ValueType>(valueVariant, &valid);
  if (valid)
  {
    this->InsertValue(valueIdx, value);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkTypeBool svtkGenericDataArray<DerivedT, ValueTypeT>::Allocate(
  svtkIdType size, svtkIdType svtkNotUsed(ext))
{
  // Allocator must updated this->Size and this->MaxId properly.
  this->MaxId = -1;
  if (size > this->Size || size == 0)
  {
    this->Size = 0;

    // let's keep the size an integral multiple of the number of components.
    size = size < 0 ? 0 : size;
    int numComps = this->GetNumberOfComponents() > 0 ? this->GetNumberOfComponents() : 1;
    double ceilNum = ceil(static_cast<double>(size) / static_cast<double>(numComps));
    svtkIdType numTuples = static_cast<svtkIdType>(ceilNum);
    // NOTE: if numTuples is 0, AllocateTuples is expected to release the
    // memory.
    if (this->AllocateTuples(numTuples) == false)
    {
      svtkErrorMacro(
        "Unable to allocate " << size << " elements of size " << sizeof(ValueType) << " bytes. ");
#if !defined NDEBUG
      // We're debugging, crash here preserving the stack
      abort();
#elif !defined SVTK_DONT_THROW_BAD_ALLOC
      // We can throw something that has universal meaning
      throw std::bad_alloc();
#else
      // We indicate that alloc failed by return
      return 0;
#endif
    }
    this->Size = numTuples * numComps;
  }
  this->DataChanged();
  return 1;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkTypeBool svtkGenericDataArray<DerivedT, ValueTypeT>::Resize(svtkIdType numTuples)
{
  int numComps = this->GetNumberOfComponents();
  svtkIdType curNumTuples = this->Size / (numComps > 0 ? numComps : 1);
  if (numTuples > curNumTuples)
  {
    // Requested size is bigger than current size.  Allocate enough
    // memory to fit the requested size and be more than double the
    // currently allocated memory.
    numTuples = curNumTuples + numTuples;
  }
  else if (numTuples == curNumTuples)
  {
    return 1;
  }
  else
  {
    // Requested size is smaller than current size.  Squeeze the
    // memory.
    this->DataChanged();
  }

  assert(numTuples >= 0);

  if (!this->ReallocateTuples(numTuples))
  {
    svtkErrorMacro("Unable to allocate " << numTuples * numComps << " elements of size "
                                        << sizeof(ValueType) << " bytes. ");
#if !defined NDEBUG
    // We're debugging, crash here preserving the stack
    abort();
#elif !defined SVTK_DONT_THROW_BAD_ALLOC
    // We can throw something that has universal meaning
    throw std::bad_alloc();
#else
    // We indicate that malloc failed by return
    return 0;
#endif
  }

  // Allocation was successful. Save it.
  this->Size = numTuples * numComps;

  // Update MaxId if we truncated:
  if ((this->Size - 1) < this->MaxId)
  {
    this->MaxId = (this->Size - 1);
  }

  return 1;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetNumberOfComponents(int num)
{
  this->svtkDataArray::SetNumberOfComponents(num);
  this->LegacyTuple.resize(num);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetNumberOfTuples(svtkIdType number)
{
  svtkIdType newSize = number * this->NumberOfComponents;
  if (this->Allocate(newSize, 0))
  {
    this->MaxId = newSize - 1;
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::Initialize()
{
  this->Resize(0);
  this->DataChanged();
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::Squeeze()
{
  this->Resize(this->GetNumberOfTuples());
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::SetTuple(
  svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other = svtkArrayDownCast<DerivedT>(source);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::SetTuple(dstTupleIdx, srcTupleIdx, source);
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (source->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << source->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  for (int c = 0; c < numComps; ++c)
  {
    this->SetTypedComponent(dstTupleIdx, c, other->GetTypedComponent(srcTupleIdx, c));
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTuples(
  svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other = svtkArrayDownCast<DerivedT>(source);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::InsertTuples(dstIds, srcIds, source);
    return;
  }

  if (dstIds->GetNumberOfIds() == 0)
  {
    return;
  }

  if (dstIds->GetNumberOfIds() != srcIds->GetNumberOfIds())
  {
    svtkErrorMacro("Mismatched number of tuples ids. Source: "
      << srcIds->GetNumberOfIds() << " Dest: " << dstIds->GetNumberOfIds());
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << other->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  svtkIdType maxSrcTupleId = srcIds->GetId(0);
  svtkIdType maxDstTupleId = dstIds->GetId(0);
  for (int i = 0; i < dstIds->GetNumberOfIds(); ++i)
  {
    // parenthesis around std::max prevent MSVC macro replacement when
    // inlined:
    maxSrcTupleId = (std::max)(maxSrcTupleId, srcIds->GetId(i));
    maxDstTupleId = (std::max)(maxDstTupleId, dstIds->GetId(i));
  }

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

  // parenthesis around std::max prevent MSVC macro replacement when
  // inlined:
  this->MaxId = (std::max)(this->MaxId, newSize - 1);

  svtkIdType numTuples = srcIds->GetNumberOfIds();
  for (svtkIdType t = 0; t < numTuples; ++t)
  {
    svtkIdType srcT = srcIds->GetId(t);
    svtkIdType dstT = dstIds->GetId(t);
    for (int c = 0; c < numComps; ++c)
    {
      this->SetTypedComponent(dstT, c, other->GetTypedComponent(srcT, c));
    }
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTuple(
  svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  this->EnsureAccessToTuple(i);
  this->SetTuple(i, j, source);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTuple(svtkIdType i, const float* source)
{
  this->EnsureAccessToTuple(i);
  this->SetTuple(i, source);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTuple(svtkIdType i, const double* source)
{
  this->EnsureAccessToTuple(i);
  this->SetTuple(i, source);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertComponent(
  svtkIdType tupleIdx, int compIdx, double value)
{
  // Update MaxId to the inserted component (not the complete tuple) for
  // compatibility with InsertNextValue.
  svtkIdType newMaxId = tupleIdx * this->NumberOfComponents + compIdx;
  if (newMaxId < this->MaxId)
  {
    newMaxId = this->MaxId;
  }
  this->EnsureAccessToTuple(tupleIdx);
  assert("Sufficient space allocated." && this->MaxId >= newMaxId);
  this->MaxId = newMaxId;
  this->SetComponent(tupleIdx, compIdx, value);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::InsertNextTuple(
  svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  svtkIdType nextTuple = this->GetNumberOfTuples();
  this->InsertTuple(nextTuple, srcTupleIdx, source);
  return nextTuple;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::InsertNextTuple(const float* tuple)
{
  svtkIdType nextTuple = this->GetNumberOfTuples();
  this->InsertTuple(nextTuple, tuple);
  return nextTuple;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::InsertNextTuple(const double* tuple)
{
  svtkIdType nextTuple = this->GetNumberOfTuples();
  this->InsertTuple(nextTuple, tuple);
  return nextTuple;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::GetTuples(
  svtkIdList* tupleIds, svtkAbstractArray* output)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other = svtkArrayDownCast<DerivedT>(output);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::GetTuples(tupleIds, output);
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components for input and output do not match.\n"
                  "Source: "
      << this->GetNumberOfComponents()
      << "\n"
         "Destination: "
      << other->GetNumberOfComponents());
    return;
  }

  svtkIdType* srcTuple = tupleIds->GetPointer(0);
  svtkIdType* srcTupleEnd = tupleIds->GetPointer(tupleIds->GetNumberOfIds());
  svtkIdType dstTuple = 0;

  while (srcTuple != srcTupleEnd)
  {
    for (int c = 0; c < numComps; ++c)
    {
      other->SetTypedComponent(dstTuple, c, this->GetTypedComponent(*srcTuple, c));
    }
    ++srcTuple;
    ++dstTuple;
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::GetTuples(
  svtkIdType p1, svtkIdType p2, svtkAbstractArray* output)
{
  // First, check for the common case of typeid(source) == typeid(this). This
  // way we don't waste time redoing the other checks in the superclass, and
  // can avoid doing a dispatch for the most common usage of this method.
  DerivedT* other = svtkArrayDownCast<DerivedT>(output);
  if (!other)
  {
    // Let the superclass handle dispatch/fallback.
    this->Superclass::GetTuples(p1, p2, output);
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (other->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components for input and output do not match.\n"
                  "Source: "
      << this->GetNumberOfComponents()
      << "\n"
         "Destination: "
      << other->GetNumberOfComponents());
    return;
  }

  // p1-p2 are inclusive
  for (svtkIdType srcT = p1, dstT = 0; srcT <= p2; ++srcT, ++dstT)
  {
    for (int c = 0; c < numComps; ++c)
    {
      other->SetTypedComponent(dstT, c, this->GetTypedComponent(srcT, c));
    }
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkArrayIterator* svtkGenericDataArray<DerivedT, ValueTypeT>::NewIterator()
{
  svtkWarningMacro(<< "No svtkArrayIterator defined for " << this->GetClassName() << " arrays.");
  return nullptr;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::InsertNextValue(ValueType value)
{
  svtkIdType nextValueIdx = this->MaxId + 1;
  if (nextValueIdx >= this->Size)
  {
    svtkIdType tuple = nextValueIdx / this->NumberOfComponents;
    this->EnsureAccessToTuple(tuple);
    // Since EnsureAccessToTuple will update the MaxId to point to the last
    // component in the last tuple, we move it back to support this method on
    // multi-component arrays.
    this->MaxId = nextValueIdx;
  }

  // Extending array without needing to reallocate:
  if (this->MaxId < nextValueIdx)
  {
    this->MaxId = nextValueIdx;
  }

  this->SetValue(nextValueIdx, value);
  return nextValueIdx;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertValue(svtkIdType valueIdx, ValueType value)
{
  svtkIdType tuple = valueIdx / this->NumberOfComponents;
  // Update MaxId to the inserted component (not the complete tuple) for
  // compatibility with InsertNextValue.
  svtkIdType newMaxId = valueIdx > this->MaxId ? valueIdx : this->MaxId;
  if (this->EnsureAccessToTuple(tuple))
  {
    assert("Sufficient space allocated." && this->MaxId >= newMaxId);
    this->MaxId = newMaxId;
    this->SetValue(valueIdx, value);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTypedTuple(
  svtkIdType tupleIdx, const ValueType* t)
{
  if (this->EnsureAccessToTuple(tupleIdx))
  {
    this->SetTypedTuple(tupleIdx, t);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkIdType svtkGenericDataArray<DerivedT, ValueTypeT>::InsertNextTypedTuple(const ValueType* t)
{
  svtkIdType nextTuple = this->GetNumberOfTuples();
  this->InsertTypedTuple(nextTuple, t);
  return nextTuple;
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::InsertTypedComponent(
  svtkIdType tupleIdx, int compIdx, ValueType val)
{
  // Update MaxId to the inserted component (not the complete tuple) for
  // compatibility with InsertNextValue.
  svtkIdType newMaxId = tupleIdx * this->NumberOfComponents + compIdx;
  if (this->MaxId > newMaxId)
  {
    newMaxId = this->MaxId;
  }
  this->EnsureAccessToTuple(tupleIdx);
  assert("Sufficient space allocated." && this->MaxId >= newMaxId);
  this->MaxId = newMaxId;
  this->SetTypedComponent(tupleIdx, compIdx, val);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::GetValueRange(ValueType range[2], int comp)
{
  this->ComputeValueRange(range, comp);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
typename svtkGenericDataArray<DerivedT, ValueTypeT>::ValueType*
svtkGenericDataArray<DerivedT, ValueTypeT>::GetValueRange(int comp)
{
  this->LegacyValueRange.resize(2);
  this->GetValueRange(this->LegacyValueRange.data(), comp);
  return &this->LegacyValueRange[0];
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::GetFiniteValueRange(ValueType range[2], int comp)
{
  this->ComputeFiniteValueRange(range, comp);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
typename svtkGenericDataArray<DerivedT, ValueTypeT>::ValueType*
svtkGenericDataArray<DerivedT, ValueTypeT>::GetFiniteValueRange(int comp)
{
  this->LegacyValueRange.resize(2);
  this->GetFiniteValueRange(this->LegacyValueRange.data(), comp);
  return &this->LegacyValueRange[0];
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::FillTypedComponent(int compIdx, ValueType value)
{
  if (compIdx < 0 || compIdx >= this->NumberOfComponents)
  {
    svtkErrorMacro(<< "Specified component " << compIdx << " is not in [0, "
                  << this->NumberOfComponents << ")");
    return;
  }
  for (svtkIdType i = 0; i < this->GetNumberOfTuples(); ++i)
  {
    this->SetTypedComponent(i, compIdx, value);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::FillValue(ValueType value)
{
  for (int i = 0; i < this->NumberOfComponents; ++i)
  {
    this->FillTypedComponent(i, value);
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::FillComponent(int compIdx, double value)
{
  this->FillTypedComponent(compIdx, static_cast<ValueType>(value));
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkGenericDataArray<DerivedT, ValueTypeT>::svtkGenericDataArray()
{
  // Initialize internal data structures:
  this->Lookup.SetArray(this);
  this->SetNumberOfComponents(this->NumberOfComponents);
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
svtkGenericDataArray<DerivedT, ValueTypeT>::~svtkGenericDataArray()
{
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::EnsureAccessToTuple(svtkIdType tupleIdx)
{
  if (tupleIdx < 0)
  {
    return false;
  }
  svtkIdType minSize = (1 + tupleIdx) * this->NumberOfComponents;
  svtkIdType expectedMaxId = minSize - 1;
  if (this->MaxId < expectedMaxId)
  {
    if (this->Size < minSize)
    {
      if (!this->Resize(tupleIdx + 1))
      {
        return false;
      }
    }
    this->MaxId = expectedMaxId;
  }
  return true;
}

// The following introduces a layer of indirection that allows us to use the
// optimized range computation logic in svtkDataArrayPrivate.txx for common
// arrays, but fallback to computing the range at double precision and then
// converting to the valuetype for unknown array types, or for types where
// the conversion from double->ValueType doesn't lose precision.

template <typename ValueType>
class svtkAOSDataArrayTemplate;
template <typename ValueType>
class svtkSOADataArrayTemplate;

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
template <typename ValueType>
class svtkScaledSOADataArrayTemplate;
#endif

namespace svtk_GDA_detail
{

// Arrays templates with compiled-in support for value ranges in
// svtkGenericDataArray.cxx
template <typename ArrayType>
struct ATIsSupported : public std::false_type
{
};

template <typename ValueType>
struct ATIsSupported<svtkAOSDataArrayTemplate<ValueType> > : public std::true_type
{
};

template <typename ValueType>
struct ATIsSupported<svtkSOADataArrayTemplate<ValueType> > : public std::true_type
{
};

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
template <typename ValueType>
struct ATIsSupported<svtkScaledSOADataArrayTemplate<ValueType> > : public std::true_type
{
};
#endif

// ValueTypes with compiled-in support for value ranges in
// svtkGenericDataArray.cxx
template <typename ValueType>
struct VTIsSupported : public std::false_type
{
};
template <>
struct VTIsSupported<long> : public std::true_type
{
};
template <>
struct VTIsSupported<unsigned long> : public std::true_type
{
};
template <>
struct VTIsSupported<long long> : public std::true_type
{
};
template <>
struct VTIsSupported<unsigned long long> : public std::true_type
{
};

// Full array types with compiled-in support for value ranges in
// svtkGenericDataArray.cxx
template <typename ArrayType, typename ValueType>
struct IsSupported
  : public std::integral_constant<bool,
      (ATIsSupported<ArrayType>::value && VTIsSupported<ValueType>::value)>
{
};

} // end namespace svtk_GDA_detail

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeValueRange(ValueType range[2], int comp)
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;

  // For array / value types without specific implementations compiled into
  // svtkGenericDataArray.cxx, fall back to the GetRange computations in
  // svtkDataArray. In these cases, either a) the ValueType's full range is
  // expressable as a double, or b) we aren't aware of the array type.
  // This reduces the number of specialized range implementations we need to
  // compile, and is also faster since we're able to cache the GetValue
  // computation (See #17666).
  if (!Supported::value)
  {
    double tmpRange[2];
    this->ComputeRange(tmpRange, comp);
    range[0] = static_cast<ValueType>(tmpRange[0]);
    range[1] = static_cast<ValueType>(tmpRange[1]);
    return;
  }

  range[0] = svtkTypeTraits<ValueType>::Max();
  range[1] = svtkTypeTraits<ValueType>::Min();

  if (comp > this->NumberOfComponents)
  {
    return;
  }

  if (comp < 0 && this->NumberOfComponents == 1)
  {
    comp = 0;
  }

  // TODO this should eventually cache the results, but we do not have support
  // for all of the information keys we need to cover all possible value types.
  if (comp < 0)
  {
    this->ComputeVectorValueRange(range);
  }
  else
  {
    this->LegacyValueRangeFull.resize(this->NumberOfComponents * 2);
    if (this->ComputeScalarValueRange(this->LegacyValueRangeFull.data()))
    {
      range[0] = this->LegacyValueRangeFull[comp * 2];
      range[1] = this->LegacyValueRangeFull[comp * 2 + 1];
    }
  }
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
void svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeFiniteValueRange(
  ValueType range[2], int comp)
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;

  // For array / value types without specific implementations compiled into
  // svtkGenericDataArray.cxx, fall back to the GetRange computations in
  // svtkDataArray. In these cases, either a) the ValueType's full range is
  // expressable as a double, or b) we aren't aware of the array type.
  // This reduces the number of specialized range implementations we need to
  // compile, and is also faster since we're able to cache the GetValue
  // computation (See #17666).
  if (!Supported::value)
  {
    double tmpRange[2];
    this->ComputeFiniteRange(tmpRange, comp);
    range[0] = static_cast<ValueType>(tmpRange[0]);
    range[1] = static_cast<ValueType>(tmpRange[1]);
    return;
  }

  range[0] = svtkTypeTraits<ValueType>::Max();
  range[1] = svtkTypeTraits<ValueType>::Min();

  if (comp > this->NumberOfComponents)
  {
    return;
  }

  if (comp < 0 && this->NumberOfComponents == 1)
  {
    comp = 0;
  }

  // TODO this should eventually cache the results, but we do not have support
  // for all of the information keys we need to cover all possible value types.
  if (comp < 0)
  {
    this->ComputeFiniteVectorValueRange(range);
  }
  else
  {
    this->LegacyValueRangeFull.resize(this->NumberOfComponents * 2);
    if (this->ComputeFiniteScalarValueRange(this->LegacyValueRangeFull.data()))
    {
      range[0] = this->LegacyValueRangeFull[comp * 2];
      range[1] = this->LegacyValueRangeFull[comp * 2 + 1];
    }
  }
}

namespace svtk_GDA_detail
{

template <typename ArrayType, typename ValueType, typename Tag>
bool ComputeScalarValueRangeImpl(ArrayType* array, ValueType* range, Tag tag, std::true_type)
{
  return ::svtkDataArrayPrivate::DoComputeScalarRange(array, range, tag);
}

template <typename ArrayType, typename ValueType, typename Tag>
bool ComputeScalarValueRangeImpl(ArrayType* array, ValueType* range, Tag tag, std::false_type)
{
  // Compute the range at double precision.
  std::size_t numComps = static_cast<size_t>(array->GetNumberOfComponents());
  std::vector<double> tmpRange(numComps * 2);
  if (!::svtkDataArrayPrivate::DoComputeScalarRange(
        static_cast<svtkDataArray*>(array), tmpRange.data(), tag))
  {
    return false;
  }

  for (std::size_t i = 0; i < numComps * 2; ++i)
  {
    range[i] = static_cast<ValueType>(tmpRange[i]);
  }

  return true;
}

template <typename ArrayType, typename ValueType, typename Tag>
bool ComputeVectorValueRangeImpl(ArrayType* array, ValueType range[2], Tag tag, std::true_type)
{
  return ::svtkDataArrayPrivate::DoComputeVectorRange(array, range, tag);
}

template <typename ArrayType, typename ValueType, typename Tag>
bool ComputeVectorValueRangeImpl(ArrayType* array, ValueType range[2], Tag tag, std::false_type)
{
  // Compute the range at double precision.
  double tmpRange[2];
  if (!::svtkDataArrayPrivate::DoComputeVectorRange(
        static_cast<svtkDataArray*>(array), tmpRange, tag))
  {
    return false;
  }

  range[0] = static_cast<ValueType>(tmpRange[0]);
  range[1] = static_cast<ValueType>(tmpRange[1]);

  return true;
}

} // namespace svtk_GDA_detail

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeScalarValueRange(ValueType* ranges)
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;
  return ComputeScalarValueRangeImpl(
    static_cast<DerivedT*>(this), ranges, svtkDataArrayPrivate::AllValues{}, Supported{});
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeVectorValueRange(ValueType range[2])
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;
  return ComputeVectorValueRangeImpl(
    static_cast<DerivedT*>(this), range, svtkDataArrayPrivate::AllValues{}, Supported{});
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeFiniteScalarValueRange(ValueType* range)
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;
  return ComputeScalarValueRangeImpl(
    static_cast<DerivedT*>(this), range, svtkDataArrayPrivate::FiniteValues{}, Supported{});
}

//-----------------------------------------------------------------------------
template <class DerivedT, class ValueTypeT>
bool svtkGenericDataArray<DerivedT, ValueTypeT>::ComputeFiniteVectorValueRange(ValueType range[2])
{
  using namespace svtk_GDA_detail;
  using Supported = IsSupported<DerivedT, ValueTypeT>;
  return ComputeVectorValueRangeImpl(
    static_cast<DerivedT*>(this), range, svtkDataArrayPrivate::FiniteValues{}, Supported{});
}

#undef svtkGenericDataArrayT

#endif // header guard

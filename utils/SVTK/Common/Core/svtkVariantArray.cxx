/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantArray.cxx

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

// We do not provide a definition for the copy constructor or
// operator=.  Block the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4661)
#endif

#include "svtkVariantArray.h"

#include "svtkArrayIteratorTemplate.h"
#include "svtkDataArray.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkSortDataArray.h"
#include "svtkStringArray.h"
#include "svtkVariant.h"

#include <algorithm>
#include <map>
#include <utility>

// Map containing updates to a svtkVariantArray that have occurred
// since we last build the svtkVariantArrayLookup.
typedef std::multimap<svtkVariant, svtkIdType, svtkVariantLessThan> svtkVariantCachedUpdates;

namespace
{
auto DefaultDeleteFunction = [](void* ptr) { delete[] reinterpret_cast<svtkVariant*>(ptr); };
}

//----------------------------------------------------------------------------
class svtkVariantArrayLookup
{
public:
  svtkVariantArrayLookup()
    : Rebuild(true)
  {
    this->SortedArray = nullptr;
    this->IndexArray = nullptr;
  }
  ~svtkVariantArrayLookup()
  {
    if (this->SortedArray)
    {
      this->SortedArray->Delete();
      this->SortedArray = nullptr;
    }
    if (this->IndexArray)
    {
      this->IndexArray->Delete();
      this->IndexArray = nullptr;
    }
  }
  svtkVariantArray* SortedArray;
  svtkIdList* IndexArray;
  svtkVariantCachedUpdates CachedUpdates;
  bool Rebuild;
};

//
// Standard functions
//

svtkStandardNewMacro(svtkVariantArray);
//----------------------------------------------------------------------------
void svtkVariantArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->Array)
  {
    os << indent << "Array: " << this->Array << "\n";
  }
  else
  {
    os << indent << "Array: (null)\n";
  }
}

//----------------------------------------------------------------------------
svtkVariantArray::svtkVariantArray()
{
  this->Array = nullptr;
  this->DeleteFunction = DefaultDeleteFunction;
  this->Lookup = nullptr;
}

//----------------------------------------------------------------------------
svtkVariantArray::~svtkVariantArray()
{
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }
  delete this->Lookup;
}

//
//
// Functions required by svtkAbstractArray
//
//

//----------------------------------------------------------------------------
svtkTypeBool svtkVariantArray::Allocate(svtkIdType sz, svtkIdType)
{
  if (sz > this->Size)
  {
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }

    this->Size = (sz > 0 ? sz : 1);
    this->Array = new svtkVariant[this->Size];
    if (!this->Array)
    {
      return 0;
    }
    this->DeleteFunction = DefaultDeleteFunction;
  }

  this->MaxId = -1;
  this->DataChanged();

  return 1;
}

//----------------------------------------------------------------------------
void svtkVariantArray::Initialize()
{
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }
  this->Array = nullptr;
  this->Size = 0;
  this->MaxId = -1;
  this->DeleteFunction = DefaultDeleteFunction;
  this->DataChanged();
}

//----------------------------------------------------------------------------
int svtkVariantArray::GetDataType() const
{
  return SVTK_VARIANT;
}

//----------------------------------------------------------------------------
int svtkVariantArray::GetDataTypeSize() const
{
  return static_cast<int>(sizeof(svtkVariant));
}

//----------------------------------------------------------------------------
int svtkVariantArray::GetElementComponentSize() const
{
  return this->GetDataTypeSize();
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetNumberOfTuples(svtkIdType number)
{
  this->SetNumberOfValues(this->NumberOfComponents * number);
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  if (source->IsA("svtkVariantArray"))
  {
    svtkVariantArray* a = svtkArrayDownCast<svtkVariantArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->SetValue(loci + cur, a->GetValue(locj + cur));
    }
  }
  else if (source->IsA("svtkDataArray"))
  {
    svtkDataArray* a = svtkArrayDownCast<svtkDataArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      // TODO : This just makes a double variant by default.
      //        We really should make the appropriate type of variant
      //        based on the subclass of svtkDataArray.
      svtkIdType tuple = (locj + cur) / a->GetNumberOfComponents();
      int component = static_cast<int>((locj + cur) % a->GetNumberOfComponents());
      this->SetValue(loci + cur, svtkVariant(a->GetComponent(tuple, component)));
    }
  }
  else if (source->IsA("svtkStringArray"))
  {
    svtkStringArray* a = svtkArrayDownCast<svtkStringArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->SetValue(loci + cur, svtkVariant(a->GetValue(locj + cur)));
    }
  }
  else
  {
    svtkWarningMacro("Unrecognized type is incompatible with svtkVariantArray.");
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  if (source->IsA("svtkVariantArray"))
  {
    svtkVariantArray* a = svtkArrayDownCast<svtkVariantArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->InsertValue(loci + cur, a->GetValue(locj + cur));
    }
  }
  else if (source->IsA("svtkDataArray"))
  {
    svtkDataArray* a = svtkArrayDownCast<svtkDataArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      svtkIdType tuple = (locj + cur) / a->GetNumberOfComponents();
      int component = static_cast<int>((locj + cur) % a->GetNumberOfComponents());
      this->InsertValue(loci + cur, svtkVariant(a->GetComponent(tuple, component)));
    }
  }
  else if (source->IsA("svtkStringArray"))
  {
    svtkStringArray* a = svtkArrayDownCast<svtkStringArray>(source);
    svtkIdType loci = i * this->NumberOfComponents;
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->InsertValue(loci + cur, svtkVariant(a->GetValue(locj + cur)));
    }
  }
  else
  {
    svtkWarningMacro("Unrecognized type is incompatible with svtkVariantArray.");
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{

  if (this->NumberOfComponents != source->GetNumberOfComponents())
  {
    svtkWarningMacro("Input and output component sizes do not match.");
    return;
  }

  svtkIdType numIds = dstIds->GetNumberOfIds();
  if (srcIds->GetNumberOfIds() != numIds)
  {
    svtkWarningMacro("Input and output id array sizes do not match.");
    return;
  }

  if (svtkVariantArray* va = svtkArrayDownCast<svtkVariantArray>(source))
  {
    for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
    {
      svtkIdType numComp = this->NumberOfComponents;
      svtkIdType srcLoc = srcIds->GetId(idIndex) * this->NumberOfComponents;
      svtkIdType dstLoc = dstIds->GetId(idIndex) * this->NumberOfComponents;
      while (numComp-- > 0)
      {
        this->InsertValue(dstLoc++, va->GetValue(srcLoc++));
      }
    }
  }
  else if (svtkDataArray* da = svtkDataArray::FastDownCast(source))
  {
    for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
    {
      svtkIdType numComp = this->NumberOfComponents;
      svtkIdType srcLoc = srcIds->GetId(idIndex) * this->NumberOfComponents;
      svtkIdType dstLoc = dstIds->GetId(idIndex) * this->NumberOfComponents;
      while (numComp-- > 0)
      {
        this->InsertValue(dstLoc++, da->GetVariantValue(srcLoc++));
      }
    }
  }
  else if (svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source))
  {
    for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
    {
      svtkIdType numComp = this->NumberOfComponents;
      svtkIdType srcLoc = srcIds->GetId(idIndex) * this->NumberOfComponents;
      svtkIdType dstLoc = dstIds->GetId(idIndex) * this->NumberOfComponents;
      while (numComp-- > 0)
      {
        this->InsertValue(dstLoc++, sa->GetVariantValue(srcLoc++));
      }
    }
  }
  else
  {
    svtkWarningMacro("Unrecognized type is incompatible with svtkVariantArray.");
  }
  this->DataChanged();
}

//------------------------------------------------------------------------------
void svtkVariantArray::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  if (this->NumberOfComponents != source->GetNumberOfComponents())
  {
    svtkWarningMacro("Input and output component sizes do not match.");
    return;
  }

  svtkIdType srcEnd = srcStart + n;
  if (srcEnd > source->GetNumberOfTuples())
  {
    svtkWarningMacro("Source range exceeds array size (srcStart="
      << srcStart << ", n=" << n << ", numTuples=" << source->GetNumberOfTuples() << ").");
    return;
  }

  for (svtkIdType i = 0; i < n; ++i)
  {
    svtkIdType numComp = this->NumberOfComponents;
    svtkIdType srcLoc = (srcStart + i) * this->NumberOfComponents;
    svtkIdType dstLoc = (dstStart + i) * this->NumberOfComponents;
    while (numComp-- > 0)
    {
      this->InsertValue(dstLoc++, source->GetVariantValue(srcLoc++));
    }
  }

  this->DataChanged();
}

//----------------------------------------------------------------------------
svtkIdType svtkVariantArray::InsertNextTuple(svtkIdType j, svtkAbstractArray* source)
{
  if (source->IsA("svtkVariantArray"))
  {
    svtkVariantArray* a = svtkArrayDownCast<svtkVariantArray>(source);
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->InsertNextValue(a->GetValue(locj + cur));
    }
  }
  else if (source->IsA("svtkDataArray"))
  {
    svtkDataArray* a = svtkArrayDownCast<svtkDataArray>(source);
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      svtkIdType tuple = (locj + cur) / a->GetNumberOfComponents();
      int component = static_cast<int>((locj + cur) % a->GetNumberOfComponents());
      this->InsertNextValue(svtkVariant(a->GetComponent(tuple, component)));
    }
  }
  else if (source->IsA("svtkStringArray"))
  {
    svtkStringArray* a = svtkArrayDownCast<svtkStringArray>(source);
    svtkIdType locj = j * a->GetNumberOfComponents();
    for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
    {
      this->InsertNextValue(svtkVariant(a->GetValue(locj + cur)));
    }
  }
  else
  {
    svtkWarningMacro("Unrecognized type is incompatible with svtkVariantArray.");
    return -1;
  }

  this->DataChanged();
  return (this->GetNumberOfTuples() - 1);
}

//----------------------------------------------------------------------------
void* svtkVariantArray::GetVoidPointer(svtkIdType id)
{
  return this->GetPointer(id);
}

//----------------------------------------------------------------------------
void svtkVariantArray::DeepCopy(svtkAbstractArray* aa)
{
  // Do nothing on a nullptr input.
  if (!aa)
  {
    return;
  }

  // Avoid self-copy.
  if (this == aa)
  {
    return;
  }

  // If data type does not match, we can't copy.
  if (aa->GetDataType() != this->GetDataType())
  {
    svtkErrorMacro(<< "Incompatible types: tried to copy an array of type "
                  << aa->GetDataTypeAsString() << " into a variant array ");
    return;
  }

  svtkVariantArray* va = svtkArrayDownCast<svtkVariantArray>(aa);
  if (va == nullptr)
  {
    svtkErrorMacro(<< "Shouldn't Happen: Couldn't downcast array into a svtkVariantArray.");
    return;
  }

  // Free our previous memory.
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }

  // Copy the given array into new memory.
  this->MaxId = va->GetMaxId();
  this->Size = va->GetSize();
  this->DeleteFunction = DefaultDeleteFunction;
  this->Array = new svtkVariant[this->Size];

  for (int i = 0; i < (this->MaxId + 1); ++i)
  {
    this->Array[i] = va->Array[i];
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::InterpolateTuple(
  svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights)
{
  // Note: Something much more fancy could be done here, allowing
  // the source array be any data type.
  if (this->GetDataType() != source->GetDataType())
  {
    svtkErrorMacro("Cannot CopyValue from array of type " << source->GetDataTypeAsString());
    return;
  }

  if (ptIndices->GetNumberOfIds() == 0)
  {
    // nothing to do.
    return;
  }

  // We use nearest neighbour for interpolating variants.
  // First determine which is the nearest neighbour using the weights-
  // it's the index with maximum weight.
  svtkIdType nearest = ptIndices->GetId(0);
  double max_weight = weights[0];
  for (int k = 1; k < ptIndices->GetNumberOfIds(); k++)
  {
    if (weights[k] > max_weight)
    {
      nearest = k;
    }
  }

  this->InsertTuple(i, nearest, source);
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1,
  svtkIdType id2, svtkAbstractArray* source2, double t)
{
  // Note: Something much more fancy could be done here, allowing
  // the source array to be any data type.
  if (source1->GetDataType() != SVTK_VARIANT || source2->GetDataType() != SVTK_VARIANT)
  {
    svtkErrorMacro("All arrays to InterpolateValue() must be of same type.");
    return;
  }

  if (t >= 0.5)
  {
    // Use p2
    this->InsertTuple(i, id2, source2);
  }
  else
  {
    // Use p1.
    this->InsertTuple(i, id1, source1);
  }
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::Squeeze()
{
  this->ResizeAndExtend(this->MaxId + 1);
}

//----------------------------------------------------------------------------
svtkTypeBool svtkVariantArray::Resize(svtkIdType sz)
{
  svtkVariant* newArray;
  svtkIdType newSize = sz * this->GetNumberOfComponents();

  if (newSize == this->Size)
  {
    return 1;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return 1;
  }

  newArray = new svtkVariant[newSize];
  if (!newArray)
  {
    svtkErrorMacro(<< "Cannot allocate memory\n");
    return 0;
  }

  if (this->Array)
  {
    svtkIdType numCopy = (newSize < this->Size ? newSize : this->Size);

    for (svtkIdType i = 0; i < numCopy; ++i)
    {
      newArray[i] = this->Array[i];
    }

    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }
  }

  if (newSize < this->Size)
  {
    this->MaxId = newSize - 1;
  }
  this->Size = newSize;
  this->Array = newArray;
  this->DeleteFunction = DefaultDeleteFunction;
  this->DataChanged();
  return 1;
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetVoidArray(void* arr, svtkIdType size, int save)
{
  this->SetArray(static_cast<svtkVariant*>(arr), size, save);
  this->DataChanged();
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetVoidArray(void* arr, svtkIdType size, int save, int deleteM)
{
  this->SetArray(static_cast<svtkVariant*>(arr), size, save, deleteM);
  this->DataChanged();
}

//----------------------------------------------------------------------------
unsigned long svtkVariantArray::GetActualMemorySize() const
{
  // NOTE: Currently does not take into account the "pointed to" data.
  size_t totalSize = 0;
  size_t numPrims = static_cast<size_t>(this->GetSize());

  totalSize = numPrims * sizeof(svtkVariant);

  return static_cast<unsigned long>(ceil(static_cast<double>(totalSize) / 1024.0)); // kibibytes
}

//----------------------------------------------------------------------------
int svtkVariantArray::IsNumeric() const
{
  return 0;
}

//----------------------------------------------------------------------------
svtkArrayIterator* svtkVariantArray::NewIterator()
{
  svtkArrayIteratorTemplate<svtkVariant>* iter = svtkArrayIteratorTemplate<svtkVariant>::New();
  iter->Initialize(this);
  return iter;
}

//
//
// Additional functions
//
//

//----------------------------------------------------------------------------
svtkVariant& svtkVariantArray::GetValue(svtkIdType id) const
{
  return this->Array[id];
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetValue(svtkIdType id, svtkVariant value)
{
  this->Array[id] = value;
  this->DataElementChanged(id);
}

//----------------------------------------------------------------------------
void svtkVariantArray::InsertValue(svtkIdType id, svtkVariant value)
{
  if (id >= this->Size)
  {
    if (!this->ResizeAndExtend(id + 1))
    {
      return;
    }
  }
  this->Array[id] = value;
  if (id > this->MaxId)
  {
    this->MaxId = id;
  }
  this->DataElementChanged(id);
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetVariantValue(svtkIdType id, svtkVariant value)
{
  this->SetValue(id, value);
}

//----------------------------------------------------------------------------
void svtkVariantArray::InsertVariantValue(svtkIdType id, svtkVariant value)
{
  this->InsertValue(id, value);
}

//----------------------------------------------------------------------------
svtkIdType svtkVariantArray::InsertNextValue(svtkVariant value)
{
  this->InsertValue(++this->MaxId, value);
  this->DataElementChanged(this->MaxId);
  return this->MaxId;
}

//----------------------------------------------------------------------------
svtkVariant* svtkVariantArray::GetPointer(svtkIdType id)
{
  return this->Array + id;
}

//----------------------------------------------------------------------------
void svtkVariantArray::SetArray(svtkVariant* arr, svtkIdType size, int save, int deleteMethod)
{
  if ((this->Array) && (this->DeleteFunction))
  {
    svtkDebugMacro(<< "Deleting the array...");
    this->DeleteFunction(this->Array);
  }
  else
  {
    svtkDebugMacro(<< "Warning, array not deleted, but will point to new array.");
  }

  svtkDebugMacro(<< "Setting array to: " << arr);

  this->Array = arr;
  this->Size = size;
  this->MaxId = size - 1;

  if (save != 0)
  {
    this->DeleteFunction = nullptr;
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_DELETE || deleteMethod == SVTK_DATA_ARRAY_USER_DEFINED)
  {
    this->DeleteFunction = DefaultDeleteFunction;
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_ALIGNED_FREE)
  {
#ifdef _WIN32
    this->DeleteFunction = _aligned_free;
#else
    this->DeleteFunction = free;
#endif
  }
  else if (deleteMethod == SVTK_DATA_ARRAY_FREE)
  {
    this->DeleteFunction = free;
  }

  this->DataChanged();
}

//-----------------------------------------------------------------------------
void svtkVariantArray::SetArrayFreeFunction(void (*callback)(void*))
{
  this->DeleteFunction = callback;
}

//----------------------------------------------------------------------------
svtkVariant* svtkVariantArray::ResizeAndExtend(svtkIdType sz)
{
  svtkVariant* newArray;
  svtkIdType newSize;

  if (sz > this->Size)
  {
    // Requested size is bigger than current size.  Allocate enough
    // memory to fit the requested size and be more than double the
    // currently allocated memory.
    newSize = this->Size + sz;
  }
  else if (sz == this->Size)
  {
    // Requested size is equal to current size.  Do nothing.
    return this->Array;
  }
  else
  {
    // Requested size is smaller than current size.  Squeeze the
    // memory.
    newSize = sz;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return nullptr;
  }

  newArray = new svtkVariant[newSize];
  if (!newArray)
  {
    svtkErrorMacro("Cannot allocate memory\n");
    return nullptr;
  }

  if (this->Array)
  {
    // can't use memcpy here
    svtkIdType numCopy = (newSize < this->Size ? newSize : this->Size);
    for (svtkIdType i = 0; i < numCopy; ++i)
    {
      newArray[i] = this->Array[i];
    }
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }
  }

  if (newSize < this->Size)
  {
    this->MaxId = newSize - 1;
  }
  this->Size = newSize;
  this->Array = newArray;
  this->DeleteFunction = DefaultDeleteFunction;
  this->DataChanged();

  return this->Array;
}

//----------------------------------------------------------------------------
void svtkVariantArray::UpdateLookup()
{
  if (!this->Lookup)
  {
    this->Lookup = new svtkVariantArrayLookup();
    this->Lookup->SortedArray = svtkVariantArray::New();
    this->Lookup->IndexArray = svtkIdList::New();
  }
  if (this->Lookup->Rebuild)
  {
    int numComps = this->GetNumberOfComponents();
    svtkIdType numTuples = this->GetNumberOfTuples();
    this->Lookup->SortedArray->DeepCopy(this);
    this->Lookup->IndexArray->SetNumberOfIds(numComps * numTuples);
    for (svtkIdType i = 0; i < numComps * numTuples; i++)
    {
      this->Lookup->IndexArray->SetId(i, i);
    }
    svtkSortDataArray::Sort(this->Lookup->SortedArray, this->Lookup->IndexArray);
    this->Lookup->Rebuild = false;
    this->Lookup->CachedUpdates.clear();
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkVariantArray::LookupValue(svtkVariant value)
{
  this->UpdateLookup();

  // First look into the cached updates, to see if there were any
  // cached changes. Find an equivalent element in the set of cached
  // indices for this value. Some of the indices may have changed
  // values since the cache was built, so we need to do this equality
  // check.
  typedef svtkVariantCachedUpdates::iterator CacheIterator;
  CacheIterator cached = this->Lookup->CachedUpdates.lower_bound(value),
                cachedEnd = this->Lookup->CachedUpdates.end();
  while (cached != cachedEnd)
  {
    // Check that we are still in the same equivalence class as the
    // value.
    if (value == (*cached).first)
    {
      // Check that the value in the original array hasn't changed.
      svtkVariant currentValue = this->GetValue(cached->second);
      if (value == currentValue)
      {
        return (*cached).second;
      }
    }
    else
    {
      break;
    }

    ++cached;
  }

  // Perform a binary search of the sorted array using STL equal_range.
  int numComps = this->Lookup->SortedArray->GetNumberOfComponents();
  svtkIdType numTuples = this->Lookup->SortedArray->GetNumberOfTuples();
  svtkVariant* ptr = this->Lookup->SortedArray->GetPointer(0);
  svtkVariant* ptrEnd = ptr + numComps * numTuples;
  svtkVariant* found = std::lower_bound(ptr, ptrEnd, value, svtkVariantLessThan());

  // Find an index with a matching value. Non-matching values might
  // show up here when the underlying value at that index has been
  // changed (so the sorted array is out-of-date).
  svtkIdType offset = static_cast<svtkIdType>(found - ptr);
  while (found != ptrEnd)
  {
    // Check whether we still have a value equivalent to what we're
    // looking for.
    if (value == *found)
    {
      // Check that the value in the original array hasn't changed.
      svtkIdType index = this->Lookup->IndexArray->GetId(offset);
      svtkVariant currentValue = this->GetValue(index);
      if (value == currentValue)
      {
        return index;
      }
    }
    else
    {
      break;
    }

    ++found;
    ++offset;
  }

  return -1;
}

//----------------------------------------------------------------------------
void svtkVariantArray::LookupValue(svtkVariant value, svtkIdList* ids)
{
  this->UpdateLookup();
  ids->Reset();

  // First look into the cached updates, to see if there were any
  // cached changes. Find an equivalent element in the set of cached
  // indices for this value. Some of the indices may have changed
  // values since the cache was built, so we need to do this equality
  // check.
  typedef svtkVariantCachedUpdates::iterator CacheIterator;
  std::pair<CacheIterator, CacheIterator> cached = this->Lookup->CachedUpdates.equal_range(value);
  while (cached.first != cached.second)
  {
    // Check that the value in the original array hasn't changed.
    svtkVariant currentValue = this->GetValue(cached.first->second);
    if (cached.first->first == currentValue)
    {
      ids->InsertNextId(cached.first->second);
    }

    ++cached.first;
  }

  // Perform a binary search of the sorted array using STL equal_range.
  int numComps = this->GetNumberOfComponents();
  svtkIdType numTuples = this->GetNumberOfTuples();
  svtkVariant* ptr = this->Lookup->SortedArray->GetPointer(0);
  svtkVariant* ptrEnd = ptr + numComps * numTuples;
  std::pair<svtkVariant*, svtkVariant*> found =
    std::equal_range(ptr, ptrEnd, value, svtkVariantLessThan());

  // Add the indices of the found items to the ID list.
  svtkIdType offset = static_cast<svtkIdType>(found.first - ptr);
  while (found.first != found.second)
  {
    // Check that the value in the original array hasn't changed.
    svtkIdType index = this->Lookup->IndexArray->GetId(offset);
    svtkVariant currentValue = this->GetValue(index);
    if (*(found.first) == currentValue)
    {
      ids->InsertNextId(index);
    }

    ++found.first;
    ++offset;
  }
}

//----------------------------------------------------------------------------
void svtkVariantArray::DataChanged()
{
  if (this->Lookup)
  {
    this->Lookup->Rebuild = true;
  }
}

//----------------------------------------------------------------------------
void svtkVariantArray::DataElementChanged(svtkIdType id)
{
  if (this->Lookup)
  {
    if (this->Lookup->Rebuild)
    {
      // We're already going to rebuild the lookup table. Do nothing.
      return;
    }

    if (this->Lookup->CachedUpdates.size() > static_cast<size_t>(this->GetNumberOfTuples() / 10))
    {
      // At this point, just rebuild the full table.
      this->Lookup->Rebuild = true;
    }
    else
    {
      // Insert this change into the set of cached updates
      std::pair<const svtkVariant, svtkIdType> value(this->GetValue(id), id);
      this->Lookup->CachedUpdates.insert(value);
    }
  }
}

//----------------------------------------------------------------------------
void svtkVariantArray::ClearLookup()
{
  delete this->Lookup;
  this->Lookup = nullptr;
}

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStringArray.cxx
  Language:  C++

  Copyright 2004 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
  license for use of this work by or on behalf of the
  U.S. Government. Redistribution and use in source and binary forms, with
  or without modification, are permitted provided that this Notice and any
  statement of authorship are reproduced on all copies.

=========================================================================*/

// We do not provide a definition for the copy constructor or
// operator=.  Block the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4661)
#endif

#include "svtkStdString.h"

#include "svtkStringArray.h"

#include "svtkArrayIteratorTemplate.h"
#include "svtkCharArray.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkObjectFactory.h"
#include "svtkSortDataArray.h"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

// Map containing updates to a svtkStringArray that have occurred
// since we last build the svtkStringArrayLookup.
typedef std::multimap<svtkStdString, svtkIdType> svtkStringCachedUpdates;

namespace
{
auto DefaultDeleteFunction = [](void* ptr) { delete[] reinterpret_cast<svtkStdString*>(ptr); };
}

//-----------------------------------------------------------------------------
class svtkStringArrayLookup
{
public:
  svtkStringArrayLookup()
    : Rebuild(true)
  {
    this->SortedArray = nullptr;
    this->IndexArray = nullptr;
  }
  ~svtkStringArrayLookup()
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
  svtkStringArray* SortedArray;
  svtkIdList* IndexArray;
  svtkStringCachedUpdates CachedUpdates;
  bool Rebuild;
};

svtkStandardNewMacro(svtkStringArray);

//-----------------------------------------------------------------------------

svtkStringArray::svtkStringArray()
{
  this->Array = nullptr;
  this->DeleteFunction = DefaultDeleteFunction;
  this->Lookup = nullptr;
}

//-----------------------------------------------------------------------------

svtkStringArray::~svtkStringArray()
{
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }
  delete this->Lookup;
}

//-----------------------------------------------------------------------------
svtkArrayIterator* svtkStringArray::NewIterator()
{
  svtkArrayIteratorTemplate<svtkStdString>* iter = svtkArrayIteratorTemplate<svtkStdString>::New();
  iter->Initialize(this);
  return iter;
}

//-----------------------------------------------------------------------------
// This method lets the user specify data to be held by the array.  The
// array argument is a pointer to the data.  size is the size of
// the array supplied by the user.  Set save to 1 to keep the class
// from deleting the array when it cleans up or reallocates memory.
// The class uses the actual array provided; it does not copy the data
// from the suppled array.
void svtkStringArray::SetArray(svtkStdString* array, svtkIdType size, int save, int deleteMethod)
{
  if (this->Array && this->DeleteFunction)
  {
    svtkDebugMacro(<< "Deleting the array...");
    this->DeleteFunction(this->Array);
  }
  else
  {
    svtkDebugMacro(<< "Warning, array not deleted, but will point to new array.");
  }

  svtkDebugMacro(<< "Setting array to: " << array);

  this->Array = array;
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
void svtkStringArray::SetArrayFreeFunction(void (*callback)(void*))
{
  this->DeleteFunction = callback;
}

//-----------------------------------------------------------------------------
// Allocate memory for this array. Delete old storage only if necessary.

svtkTypeBool svtkStringArray::Allocate(svtkIdType sz, svtkIdType)
{
  if (sz > this->Size)
  {
    if (this->DeleteFunction)
    {
      this->DeleteFunction(this->Array);
    }

    this->Size = (sz > 0 ? sz : 1);
    this->Array = new svtkStdString[this->Size];
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

//-----------------------------------------------------------------------------
// Release storage and reset array to initial state.

void svtkStringArray::Initialize()
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

//-----------------------------------------------------------------------------
// Deep copy of another string array.

void svtkStringArray::DeepCopy(svtkAbstractArray* aa)
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
                  << aa->GetDataTypeAsString() << " into a string array ");
    return;
  }

  svtkStringArray* fa = svtkArrayDownCast<svtkStringArray>(aa);
  if (fa == nullptr)
  {
    svtkErrorMacro(<< "Shouldn't Happen: Couldn't downcast array into a svtkStringArray.");
    return;
  }

  // Free our previous memory.
  if (this->DeleteFunction)
  {
    this->DeleteFunction(this->Array);
  }

  // Copy the given array into new memory.
  this->MaxId = fa->GetMaxId();
  this->Size = fa->GetSize();
  this->DeleteFunction = DefaultDeleteFunction;
  this->Array = new svtkStdString[this->Size];

  for (int i = 0; i < this->Size; ++i)
  {
    this->Array[i] = fa->Array[i];
  }
  this->DataChanged();
}

//-----------------------------------------------------------------------------
// Interpolate array value from other array value given the
// indices and associated interpolation weights.
// This method assumes that the two arrays are of the same time.
void svtkStringArray::InterpolateTuple(
  svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights)
{
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

  // We use nearest neighbour for interpolating strings.
  // First determine which is the nearest neighbour using the weights-
  // it's the index with maximum weight.
  svtkIdType nearest = ptIndices->GetId(0);
  double max_weight = weights[0];
  for (int k = 1; k < ptIndices->GetNumberOfIds(); k++)
  {
    if (weights[k] > max_weight)
    {
      nearest = ptIndices->GetId(k);
      max_weight = weights[k];
    }
  }

  this->InsertTuple(i, nearest, source);
}

//-----------------------------------------------------------------------------
// Interpolate value from the two values, p1 and p2, and an
// interpolation factor, t. The interpolation factor ranges from (0,1),
// with t=0 located at p1. This method assumes that the three arrays are of
// the same type. p1 is value at index id1 in fromArray1, while, p2 is
// value at index id2 in fromArray2.
void svtkStringArray::InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1,
  svtkIdType id2, svtkAbstractArray* source2, double t)
{
  if (source1->GetDataType() != SVTK_STRING || source2->GetDataType() != SVTK_STRING)
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
}

//-----------------------------------------------------------------------------
void svtkStringArray::PrintSelf(ostream& os, svtkIndent indent)
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

//-----------------------------------------------------------------------------
// Protected function does "reallocate"

svtkStdString* svtkStringArray::ResizeAndExtend(svtkIdType sz)
{
  svtkStdString* newArray;
  svtkIdType newSize;

  if (sz > this->Size)
  {
    // Requested size is bigger than current size.  Allocate enough
    // memory to fit the requested size and be more than double the
    // currently allocated memory.
    newSize = (this->Size + 1) + sz;
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

  newArray = new svtkStdString[newSize];
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

//-----------------------------------------------------------------------------
svtkTypeBool svtkStringArray::Resize(svtkIdType sz)
{
  svtkStdString* newArray;
  svtkIdType newSize = sz;

  if (newSize == this->Size)
  {
    return 1;
  }

  if (newSize <= 0)
  {
    this->Initialize();
    return 1;
  }

  newArray = new svtkStdString[newSize];
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
      this->DeleteFunction = DefaultDeleteFunction;
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

//-----------------------------------------------------------------------------
svtkStdString* svtkStringArray::WritePointer(svtkIdType id, svtkIdType number)
{
  svtkIdType newSize = id + number;
  if (newSize > this->Size)
  {
    this->ResizeAndExtend(newSize);
  }
  if ((--newSize) > this->MaxId)
  {
    this->MaxId = newSize;
  }
  this->DataChanged();
  return this->Array + id;
}

//-----------------------------------------------------------------------------
void svtkStringArray::InsertValue(svtkIdType id, svtkStdString f)
{
  if (id >= this->Size)
  {
    if (!this->ResizeAndExtend(id + 1))
    {
      return;
    }
  }
  this->Array[id] = f;
  if (id > this->MaxId)
  {
    this->MaxId = id;
  }
  this->DataElementChanged(id);
}

//-----------------------------------------------------------------------------
svtkIdType svtkStringArray::InsertNextValue(svtkStdString f)
{
  this->InsertValue(++this->MaxId, f);
  this->DataElementChanged(this->MaxId);
  return this->MaxId;
}

// ----------------------------------------------------------------------------
int svtkStringArray::GetDataTypeSize() const
{
  return static_cast<int>(sizeof(svtkStdString));
}

// ----------------------------------------------------------------------------
unsigned long svtkStringArray::GetActualMemorySize() const
{
  size_t totalSize = 0;
  size_t numPrims = static_cast<size_t>(this->GetSize());

  for (size_t i = 0; i < numPrims; ++i)
  {
    totalSize += sizeof(svtkStdString);
    totalSize += this->Array[i].size() * sizeof(svtkStdString::value_type);
  }

  return static_cast<unsigned long>(ceil(static_cast<double>(totalSize) / 1024.0)); // kibibytes
}

// ----------------------------------------------------------------------------
svtkIdType svtkStringArray::GetDataSize() const
{
  size_t size = 0;
  size_t numStrs = static_cast<size_t>(this->GetMaxId() + 1);
  for (size_t i = 0; i < numStrs; i++)
  {
    size += this->Array[i].size() + 1;
    // (+1) for termination character.
  }
  return static_cast<svtkIdType>(size);
}

// ----------------------------------------------------------------------------
// Set the tuple at the ith location using the jth tuple in the source array.
// This method assumes that the two arrays have the same type
// and structure. Note that range checking and memory allocation is not
// performed; use in conjunction with SetNumberOfTuples() to allocate space.
void svtkStringArray::SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return;
  }

  svtkIdType loci = i * this->NumberOfComponents;
  svtkIdType locj = j * sa->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->SetValue(loci + cur, sa->GetValue(locj + cur));
  }
  this->DataChanged();
}

// ----------------------------------------------------------------------------
// Insert the jth tuple in the source array, at ith location in this array.
// Note that memory allocation is performed as necessary to hold the data.
void svtkStringArray::InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return;
  }

  svtkIdType loci = i * this->NumberOfComponents;
  svtkIdType locj = j * sa->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->InsertValue(loci + cur, sa->GetValue(locj + cur));
  }
  this->DataChanged();
}

// ----------------------------------------------------------------------------
void svtkStringArray::InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{
  svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return;
  }

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

  for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
  {
    svtkIdType numComp = this->NumberOfComponents;
    svtkIdType srcLoc = srcIds->GetId(idIndex) * this->NumberOfComponents;
    svtkIdType dstLoc = dstIds->GetId(idIndex) * this->NumberOfComponents;
    while (numComp-- > 0)
    {
      this->InsertValue(dstLoc++, sa->GetValue(srcLoc++));
    }
  }

  this->DataChanged();
}

// ----------------------------------------------------------------------------
void svtkStringArray::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return;
  }

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
      this->InsertValue(dstLoc++, sa->GetValue(srcLoc++));
    }
  }

  this->DataChanged();
}

// ----------------------------------------------------------------------------
// Insert the jth tuple in the source array, at the end in this array.
// Note that memory allocation is performed as necessary to hold the data.
// Returns the location at which the data was inserted.
svtkIdType svtkStringArray::InsertNextTuple(svtkIdType j, svtkAbstractArray* source)
{
  svtkStringArray* sa = svtkArrayDownCast<svtkStringArray>(source);
  if (!sa)
  {
    svtkWarningMacro("Input and outputs array data types do not match.");
    return -1;
  }

  svtkIdType locj = j * sa->GetNumberOfComponents();
  for (svtkIdType cur = 0; cur < this->NumberOfComponents; cur++)
  {
    this->InsertNextValue(sa->GetValue(locj + cur));
  }
  this->DataChanged();
  return (this->GetNumberOfTuples() - 1);
}

// ----------------------------------------------------------------------------
svtkStdString& svtkStringArray::GetValue(svtkIdType id)
{
  return this->Array[id];
}

// ----------------------------------------------------------------------------
void svtkStringArray::GetTuples(svtkIdList* indices, svtkAbstractArray* aa)
{
  if (aa == nullptr)
  {
    svtkErrorMacro(<< "GetTuples: Output array is null!");
    return;
  }

  svtkStringArray* output = svtkArrayDownCast<svtkStringArray>(aa);

  if (output == nullptr)
  {
    svtkErrorMacro(<< "Can't copy values from a string array into an array "
                  << "of type " << aa->GetDataTypeAsString());
    return;
  }

  for (svtkIdType i = 0; i < indices->GetNumberOfIds(); ++i)
  {
    svtkIdType index = indices->GetId(i);
    output->SetValue(i, this->GetValue(index));
  }
}

// ----------------------------------------------------------------------------
void svtkStringArray::GetTuples(svtkIdType startIndex, svtkIdType endIndex, svtkAbstractArray* aa)
{
  if (aa == nullptr)
  {
    svtkErrorMacro(<< "GetTuples: Output array is null!");
    return;
  }

  svtkStringArray* output = svtkArrayDownCast<svtkStringArray>(aa);

  if (output == nullptr)
  {
    svtkErrorMacro(<< "Can't copy values from a string array into an array "
                  << "of type " << aa->GetDataTypeAsString());
    return;
  }

  for (svtkIdType i = 0; i < (endIndex - startIndex) + 1; ++i)
  {
    svtkIdType index = startIndex + i;
    output->SetValue(i, this->GetValue(index));
  }
}

//-----------------------------------------------------------------------------
void svtkStringArray::UpdateLookup()
{
  if (!this->Lookup)
  {
    this->Lookup = new svtkStringArrayLookup();
    this->Lookup->SortedArray = svtkStringArray::New();
    this->Lookup->IndexArray = svtkIdList::New();
  }
  if (this->Lookup->Rebuild)
  {
    int numComps = this->GetNumberOfComponents();
    svtkIdType numTuples = this->GetNumberOfTuples();
    this->Lookup->SortedArray->Initialize();
    this->Lookup->SortedArray->SetNumberOfComponents(numComps);
    this->Lookup->SortedArray->SetNumberOfTuples(numTuples);
    this->Lookup->IndexArray->SetNumberOfIds(numComps * numTuples);
    std::vector<std::pair<svtkStdString, svtkIdType> > v;
    for (svtkIdType i = 0; i < numComps * numTuples; i++)
    {
      v.push_back(std::pair<svtkStdString, svtkIdType>(this->Array[i], i));
    }
    std::sort(v.begin(), v.end());
    for (svtkIdType i = 0; i < numComps * numTuples; i++)
    {
      this->Lookup->SortedArray->SetValue(i, v[i].first);
      this->Lookup->IndexArray->SetId(i, v[i].second);
    }
    this->Lookup->Rebuild = false;
    this->Lookup->CachedUpdates.clear();
  }
}

//-----------------------------------------------------------------------------
svtkIdType svtkStringArray::LookupValue(svtkVariant var)
{
  return this->LookupValue(var.ToString());
}

//-----------------------------------------------------------------------------
void svtkStringArray::LookupValue(svtkVariant var, svtkIdList* ids)
{
  this->LookupValue(var.ToString(), ids);
}

//-----------------------------------------------------------------------------
svtkIdType svtkStringArray::LookupValue(const svtkStdString& value)
{
  this->UpdateLookup();

  // First look into the cached updates, to see if there were any
  // cached changes. Find an equivalent element in the set of cached
  // indices for this value. Some of the indices may have changed
  // values since the cache was built, so we need to do this equality
  // check.
  typedef svtkStringCachedUpdates::iterator CacheIterator;
  CacheIterator cached = this->Lookup->CachedUpdates.lower_bound(value),
                cachedEnd = this->Lookup->CachedUpdates.end();
  while (cached != cachedEnd)
  {
    // Check that we are still in the same equivalence class as the
    // value.
    if (value == cached->first)
    {
      // Check that the value in the original array hasn't changed.
      svtkStdString currentValue = this->GetValue(cached->second);
      if (value == currentValue)
      {
        return cached->second;
      }
    }
    else
    {
      break;
    }

    ++cached;
  }

  int numComps = this->Lookup->SortedArray->GetNumberOfComponents();
  svtkIdType numTuples = this->Lookup->SortedArray->GetNumberOfTuples();
  svtkStdString* ptr = this->Lookup->SortedArray->GetPointer(0);
  svtkStdString* ptrEnd = ptr + numComps * numTuples;
  svtkStdString* found = std::lower_bound(ptr, ptrEnd, value);

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
      svtkStdString currentValue = this->GetValue(index);
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

//-----------------------------------------------------------------------------
void svtkStringArray::LookupValue(const svtkStdString& value, svtkIdList* ids)
{
  this->UpdateLookup();
  ids->Reset();

  // First look into the cached updates, to see if there were any
  // cached changes. Find an equivalent element in the set of cached
  // indices for this value. Some of the indices may have changed
  // values since the cache was built, so we need to do this equality
  // check.
  typedef svtkStringCachedUpdates::iterator CacheIterator;
  std::pair<CacheIterator, CacheIterator> cached = this->Lookup->CachedUpdates.equal_range(value);
  while (cached.first != cached.second)
  {
    // Check that the value in the original array hasn't changed.
    svtkStdString currentValue = this->GetValue(cached.first->second);
    if (cached.first->first == currentValue)
    {
      ids->InsertNextId(cached.first->second);
    }

    ++cached.first;
  }

  // Perform a binary search of the sorted array using STL equal_range.
  int numComps = this->GetNumberOfComponents();
  svtkIdType numTuples = this->GetNumberOfTuples();
  svtkStdString* ptr = this->Lookup->SortedArray->GetPointer(0);
  std::pair<svtkStdString*, svtkStdString*> found =
    std::equal_range(ptr, ptr + numComps * numTuples, value);

  // Add the indices of the found items to the ID list.
  svtkIdType offset = static_cast<svtkIdType>(found.first - ptr);
  while (found.first != found.second)
  {
    // Check that the value in the original array hasn't changed.
    svtkIdType index = this->Lookup->IndexArray->GetId(offset);
    svtkStdString currentValue = this->GetValue(index);
    if (*found.first == currentValue)
    {
      ids->InsertNextId(index);
    }

    ++found.first;
    ++offset;
  }
}

//-----------------------------------------------------------------------------
void svtkStringArray::DataChanged()
{
  if (this->Lookup)
  {
    this->Lookup->Rebuild = true;
  }
}

//----------------------------------------------------------------------------
void svtkStringArray::DataElementChanged(svtkIdType id)
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
      std::pair<const svtkStdString, svtkIdType> value(this->GetValue(id), id);
      this->Lookup->CachedUpdates.insert(value);
    }
  }
}

//-----------------------------------------------------------------------------
void svtkStringArray::ClearLookup()
{
  delete this->Lookup;
  this->Lookup = nullptr;
}

// ----------------------------------------------------------------------------

//
//
// Below here are interface methods to allow values to be inserted as
// const char * instead of svtkStdString.  Yes, they're trivial.  The
// wrapper code needs them.
//
//

void svtkStringArray::SetValue(svtkIdType id, const char* value)
{
  if (value)
  {
    this->SetValue(id, svtkStdString(value));
  }
}

void svtkStringArray::InsertValue(svtkIdType id, const char* value)
{
  if (value)
  {
    this->InsertValue(id, svtkStdString(value));
  }
}

void svtkStringArray::SetVariantValue(svtkIdType id, svtkVariant value)
{
  this->SetValue(id, value.ToString());
}

void svtkStringArray::InsertVariantValue(svtkIdType id, svtkVariant value)
{
  this->InsertValue(id, value.ToString());
}

svtkIdType svtkStringArray::InsertNextValue(const char* value)
{
  if (value)
  {
    return this->InsertNextValue(svtkStdString(value));
  }
  return this->MaxId;
}

svtkIdType svtkStringArray::LookupValue(const char* value)
{
  if (value)
  {
    return this->LookupValue(svtkStdString(value));
  }
  return -1;
}

void svtkStringArray::LookupValue(const char* value, svtkIdList* ids)
{
  if (value)
  {
    this->LookupValue(svtkStdString(value), ids);
    return;
  }
  ids->Reset();
}

// ----------------------------------------------------------------------------

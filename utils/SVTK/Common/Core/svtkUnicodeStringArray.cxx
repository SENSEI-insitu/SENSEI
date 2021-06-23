/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnicodeStringArray.cxx
  Language:  C++

  Copyright 2004 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
  license for use of this work by or on behalf of the
  U.S. Government. Redistribution and use in source and binary forms, with
  or without modification, are permitted provided that this Notice and any
  statement of authorship are reproduced on all copies.

=========================================================================*/

#include "svtkUnicodeString.h"

#include "svtkArrayIteratorTemplate.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkUnicodeStringArray.h"

#include <algorithm>
#include <vector>

class svtkUnicodeStringArray::Implementation
{
public:
  typedef std::vector<svtkUnicodeString> StorageT;
  StorageT Storage;
};

svtkStandardNewMacro(svtkUnicodeStringArray);

svtkUnicodeStringArray::svtkUnicodeStringArray()
{
  this->Internal = new Implementation;
}

svtkUnicodeStringArray::~svtkUnicodeStringArray()
{
  delete this->Internal;
}

void svtkUnicodeStringArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

svtkTypeBool svtkUnicodeStringArray::Allocate(svtkIdType sz, svtkIdType)
{
  this->Internal->Storage.reserve(sz);
  this->DataChanged();
  return 1;
}

void svtkUnicodeStringArray::Initialize()
{
  this->Internal->Storage.clear();
  this->DataChanged();
}

int svtkUnicodeStringArray::GetDataType() const
{
  return SVTK_UNICODE_STRING;
}

int svtkUnicodeStringArray::GetDataTypeSize() const
{
  return 0;
}

int svtkUnicodeStringArray::GetElementComponentSize() const
{
  return sizeof(svtkUnicodeString::value_type);
}

void svtkUnicodeStringArray::SetNumberOfTuples(svtkIdType number)
{
  this->Internal->Storage.resize(number);
  this->DataChanged();
}

void svtkUnicodeStringArray::SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkUnicodeStringArray* const array = svtkArrayDownCast<svtkUnicodeStringArray>(source);
  if (!array)
  {
    svtkWarningMacro("Input and output array data types do not match.");
    return;
  }

  this->Internal->Storage[i] = array->Internal->Storage[j];
  this->DataChanged();
}

void svtkUnicodeStringArray::InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source)
{
  svtkUnicodeStringArray* const array = svtkArrayDownCast<svtkUnicodeStringArray>(source);
  if (!array)
  {
    svtkWarningMacro("Input and output array data types do not match.");
    return;
  }

  if (static_cast<svtkIdType>(this->Internal->Storage.size()) <= i)
    this->Internal->Storage.resize(i + 1);

  this->Internal->Storage[i] = array->Internal->Storage[j];
  this->DataChanged();
}

void svtkUnicodeStringArray::InsertTuples(
  svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{
  svtkUnicodeStringArray* const array = svtkArrayDownCast<svtkUnicodeStringArray>(source);
  if (!array)
  {
    svtkWarningMacro("Input and output array data types do not match.");
    return;
  }

  svtkIdType numIds = dstIds->GetNumberOfIds();
  if (srcIds->GetNumberOfIds() != numIds)
  {
    svtkWarningMacro("Input and output id array sizes do not match.");
    return;
  }

  // Find maximum destination id and resize if needed
  svtkIdType maxDstId = 0;
  for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
  {
    maxDstId = std::max(maxDstId, dstIds->GetId(idIndex));
  }

  if (static_cast<svtkIdType>(this->Internal->Storage.size()) <= maxDstId)
  {
    this->Internal->Storage.resize(maxDstId + 1);
  }

  // Copy data
  for (svtkIdType idIndex = 0; idIndex < numIds; ++idIndex)
  {
    this->Internal->Storage[dstIds->GetId(idIndex)] =
      array->Internal->Storage[srcIds->GetId(idIndex)];
  }

  this->DataChanged();
}

//------------------------------------------------------------------------------
void svtkUnicodeStringArray::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  svtkUnicodeStringArray* sa = svtkArrayDownCast<svtkUnicodeStringArray>(source);
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

svtkIdType svtkUnicodeStringArray::InsertNextTuple(svtkIdType j, svtkAbstractArray* source)
{
  svtkUnicodeStringArray* const array = svtkArrayDownCast<svtkUnicodeStringArray>(source);
  if (!array)
  {
    svtkWarningMacro("Input and output array data types do not match.");
    return 0;
  }

  this->Internal->Storage.push_back(array->Internal->Storage[j]);
  this->DataChanged();
  return static_cast<svtkIdType>(this->Internal->Storage.size()) - 1;
}

void* svtkUnicodeStringArray::GetVoidPointer(svtkIdType id)
{
  // Err.. not totally sure what to do here
  if (this->Internal->Storage.empty())
    return nullptr;
  else
    return &this->Internal->Storage[id];
}

void svtkUnicodeStringArray::DeepCopy(svtkAbstractArray* da)
{
  if (!da)
    return;

  if (this == da)
    return;

  svtkUnicodeStringArray* const array = svtkArrayDownCast<svtkUnicodeStringArray>(da);
  if (!array)
  {
    svtkWarningMacro("Input and output array data types do not match.");
    return;
  }

  this->Internal->Storage = array->Internal->Storage;
  this->DataChanged();
}

void svtkUnicodeStringArray::InterpolateTuple(
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

void svtkUnicodeStringArray::InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1,
  svtkIdType id2, svtkAbstractArray* source2, double t)
{
  if (source1->GetDataType() != this->GetDataType() ||
    source2->GetDataType() != this->GetDataType())
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

void svtkUnicodeStringArray::Squeeze()
{
  this->Internal->Storage.shrink_to_fit();
  this->DataChanged();
}

svtkTypeBool svtkUnicodeStringArray::Resize(svtkIdType numTuples)
{
  this->Internal->Storage.resize(numTuples);
  this->DataChanged();
  return 1;
}

void svtkUnicodeStringArray::SetVoidArray(void*, svtkIdType, int)
{
  svtkErrorMacro("Not implemented.");
}

void svtkUnicodeStringArray::SetVoidArray(void*, svtkIdType, int, int)
{
  svtkErrorMacro("Not implemented.");
}

void svtkUnicodeStringArray::SetArrayFreeFunction(void (*)(void*))
{
  svtkErrorMacro("Not implemented.");
}

unsigned long svtkUnicodeStringArray::GetActualMemorySize() const
{
  unsigned long count = 0;
  for (Implementation::StorageT::size_type i = 0; i != this->Internal->Storage.size(); ++i)
  {
    count += static_cast<unsigned long>(this->Internal->Storage[i].byte_count());
    count += static_cast<unsigned long>(sizeof(svtkUnicodeString));
  }
  return count;
}

int svtkUnicodeStringArray::IsNumeric() const
{
  return 0;
}

svtkArrayIterator* svtkUnicodeStringArray::NewIterator()
{
  svtkErrorMacro("Not implemented.");
  return nullptr;
}

svtkVariant svtkUnicodeStringArray::GetVariantValue(svtkIdType idx)
{
  return this->GetValue(idx);
}

svtkIdType svtkUnicodeStringArray::LookupValue(svtkVariant value)
{
  const svtkUnicodeString search_value = value.ToUnicodeString();

  for (Implementation::StorageT::size_type i = 0; i != this->Internal->Storage.size(); ++i)
  {
    if (this->Internal->Storage[i] == search_value)
      return static_cast<svtkIdType>(i);
  }

  return -1;
}

void svtkUnicodeStringArray::LookupValue(svtkVariant value, svtkIdList* ids)
{
  const svtkUnicodeString search_value = value.ToUnicodeString();

  ids->Reset();
  for (Implementation::StorageT::size_type i = 0; i != this->Internal->Storage.size(); ++i)
  {
    if (this->Internal->Storage[i] == search_value)
      ids->InsertNextId(static_cast<svtkIdType>(i));
  }
}

void svtkUnicodeStringArray::SetVariantValue(svtkIdType id, svtkVariant value)
{
  this->SetValue(id, value.ToUnicodeString());
}

void svtkUnicodeStringArray::InsertVariantValue(svtkIdType id, svtkVariant value)
{
  this->InsertValue(id, value.ToUnicodeString());
}

void svtkUnicodeStringArray::DataChanged()
{
  this->MaxId = static_cast<svtkIdType>(this->Internal->Storage.size()) - 1;
}

void svtkUnicodeStringArray::ClearLookup() {}

svtkIdType svtkUnicodeStringArray::InsertNextValue(const svtkUnicodeString& value)
{
  this->Internal->Storage.push_back(value);
  this->DataChanged();
  return static_cast<svtkIdType>(this->Internal->Storage.size()) - 1;
}

void svtkUnicodeStringArray::InsertValue(svtkIdType i, const svtkUnicodeString& value)
{
  // Range check
  if (static_cast<svtkIdType>(this->Internal->Storage.size()) <= i)
    this->Internal->Storage.resize(i + 1);

  this->SetValue(i, value);
}

void svtkUnicodeStringArray::SetValue(svtkIdType i, const svtkUnicodeString& value)
{
  this->Internal->Storage[i] = value;
  this->DataChanged();
}

svtkUnicodeString& svtkUnicodeStringArray::GetValue(svtkIdType i)
{
  return this->Internal->Storage[i];
}

void svtkUnicodeStringArray::InsertNextUTF8Value(const char* value)
{
  this->InsertNextValue(svtkUnicodeString::from_utf8(value));
}

void svtkUnicodeStringArray::SetUTF8Value(svtkIdType i, const char* value)
{
  this->SetValue(i, svtkUnicodeString::from_utf8(value));
}

const char* svtkUnicodeStringArray::GetUTF8Value(svtkIdType i)
{
  return this->Internal->Storage[i].utf8_str();
}

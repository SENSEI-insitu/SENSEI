/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSparseArray.txx

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

#ifndef svtkSparseArray_txx
#define svtkSparseArray_txx

#include <algorithm>
#include <limits>

template <typename T>
svtkSparseArray<T>* svtkSparseArray<T>::New()
{
  // Don't use object factory macros on templates, it'll confuse the object
  // factory.
  svtkSparseArray<T>* ret = new svtkSparseArray<T>;
  ret->InitializeObjectBase();
  return ret;
}

template <typename T>
void svtkSparseArray<T>::PrintSelf(ostream& os, svtkIndent indent)
{
  svtkSparseArray<T>::Superclass::PrintSelf(os, indent);
}

template <typename T>
bool svtkSparseArray<T>::IsDense()
{
  return false;
}

template <typename T>
const svtkArrayExtents& svtkSparseArray<T>::GetExtents()
{
  return this->Extents;
}

template <typename T>
typename svtkSparseArray<T>::SizeT svtkSparseArray<T>::GetNonNullSize()
{
  return this->Values.size();
}

template <typename T>
void svtkSparseArray<T>::GetCoordinatesN(const SizeT n, svtkArrayCoordinates& coordinates)
{
  coordinates.SetDimensions(this->GetDimensions());
  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
    coordinates[i] = this->Coordinates[i][n];
}

template <typename T>
svtkArray* svtkSparseArray<T>::DeepCopy()
{
  ThisT* const copy = ThisT::New();

  copy->SetName(this->GetName());
  copy->Extents = this->Extents;
  copy->DimensionLabels = this->DimensionLabels;
  copy->Coordinates = this->Coordinates;
  copy->Values = this->Values;
  copy->NullValue = this->NullValue;

  return copy;
}

template <typename T>
const T& svtkSparseArray<T>::GetValue(CoordinateT i)
{
  if (1 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return this->NullValue;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    return this->Values[row];
  }

  return this->NullValue;
}

template <typename T>
const T& svtkSparseArray<T>::GetValue(CoordinateT i, CoordinateT j)
{
  if (2 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return this->NullValue;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    if (j != this->Coordinates[1][row])
      continue;

    return this->Values[row];
  }

  return this->NullValue;
}

template <typename T>
const T& svtkSparseArray<T>::GetValue(CoordinateT i, CoordinateT j, CoordinateT k)
{
  if (3 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return this->NullValue;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    if (j != this->Coordinates[1][row])
      continue;

    if (k != this->Coordinates[2][row])
      continue;

    return this->Values[row];
  }

  return this->NullValue;
}

template <typename T>
const T& svtkSparseArray<T>::GetValue(const svtkArrayCoordinates& coordinates)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return this->NullValue;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    for (DimensionT column = 0; column != this->GetDimensions(); ++column)
    {
      if (coordinates[column] != this->Coordinates[column][row])
        break;

      if (column + 1 == this->GetDimensions())
        return this->Values[row];
    }
  }

  return this->NullValue;
}

template <typename T>
const T& svtkSparseArray<T>::GetValueN(const SizeT n)
{
  return this->Values[n];
}

template <typename T>
void svtkSparseArray<T>::SetValue(CoordinateT i, const T& value)
{
  if (1 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    this->Values[row] = value;
    return;
  }

  // Element doesn't already exist, so add it to the end of the list ...
  this->AddValue(i, value);
}

template <typename T>
void svtkSparseArray<T>::SetValue(CoordinateT i, CoordinateT j, const T& value)
{
  if (2 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    if (j != this->Coordinates[1][row])
      continue;

    this->Values[row] = value;
    return;
  }

  // Element doesn't already exist, so add it to the end of the list ...
  this->AddValue(i, j, value);
}

template <typename T>
void svtkSparseArray<T>::SetValue(CoordinateT i, CoordinateT j, CoordinateT k, const T& value)
{
  if (3 != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    if (i != this->Coordinates[0][row])
      continue;

    if (j != this->Coordinates[1][row])
      continue;

    if (k != this->Coordinates[2][row])
      continue;

    this->Values[row] = value;
    return;
  }

  // Element doesn't already exist, so add it to the end of the list ...
  this->AddValue(i, j, k, value);
}

template <typename T>
void svtkSparseArray<T>::SetValue(const svtkArrayCoordinates& coordinates, const T& value)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  // Do a naive linear-search for the time-being ...
  for (svtkIdType row = 0; row != static_cast<svtkIdType>(this->Values.size()); ++row)
  {
    for (DimensionT column = 0; column != this->GetDimensions(); ++column)
    {
      if (coordinates[column] != this->Coordinates[column][row])
        break;

      if (column + 1 == this->GetDimensions())
      {
        this->Values[row] = value;
        return;
      }
    }
  }

  // Element doesn't already exist, so add it to the end of the list ...
  this->AddValue(coordinates, value);
}

template <typename T>
void svtkSparseArray<T>::SetValueN(const SizeT n, const T& value)
{
  this->Values[n] = value;
}

template <typename T>
void svtkSparseArray<T>::SetNullValue(const T& value)
{
  this->NullValue = value;
}

template <typename T>
const T& svtkSparseArray<T>::GetNullValue()
{
  return this->NullValue;
}

template <typename T>
void svtkSparseArray<T>::Clear()
{
  for (DimensionT column = 0; column != this->GetDimensions(); ++column)
    this->Coordinates[column].clear();

  this->Values.clear();
}

/// Predicate object for use with std::sort().  Given a svtkArraySort object that defines which array
/// dimensions will be sorted in what order, SortCoordinates is used to establish a sorted order for
/// the values stored in svtkSparseArray. Note that SortCoordinates never actually modifies its
/// inputs.
struct SortCoordinates
{
  SortCoordinates(const svtkArraySort& sort, const std::vector<std::vector<svtkIdType> >& coordinates)
    : Sort(&sort)
    , Coordinates(&coordinates)
  {
  }

  bool operator()(const svtkIdType lhs, const svtkIdType rhs) const
  {
    const svtkArraySort& sort = *this->Sort;
    const std::vector<std::vector<svtkIdType> >& coordinates = *this->Coordinates;

    for (svtkIdType i = 0; i != sort.GetDimensions(); ++i)
    {
      if (coordinates[sort[i]][lhs] == coordinates[sort[i]][rhs])
        continue;

      return coordinates[sort[i]][lhs] < coordinates[sort[i]][rhs];
    }

    return false;
  }

  const svtkArraySort* Sort;
  const std::vector<std::vector<svtkIdType> >* Coordinates;
};

template <typename T>
void svtkSparseArray<T>::Sort(const svtkArraySort& sort)
{
  if (sort.GetDimensions() < 1)
  {
    svtkErrorMacro(<< "Sort must order at least one dimension.");
    return;
  }

  for (DimensionT i = 0; i != sort.GetDimensions(); ++i)
  {
    if (sort[i] < 0 || sort[i] >= this->GetDimensions())
    {
      svtkErrorMacro(<< "Sort dimension out-of-bounds.");
      return;
    }
  }

  const SizeT count = this->GetNonNullSize();
  std::vector<DimensionT> sort_order(count);
  for (SizeT i = 0; i != count; ++i)
    sort_order[i] = static_cast<DimensionT>(i);
  std::sort(sort_order.begin(), sort_order.end(), SortCoordinates(sort, this->Coordinates));

  std::vector<DimensionT> temp_coordinates(count);
  for (DimensionT j = 0; j != this->GetDimensions(); ++j)
  {
    for (SizeT i = 0; i != count; ++i)
      temp_coordinates[i] = this->Coordinates[j][sort_order[i]];
    std::swap(temp_coordinates, this->Coordinates[j]);
  }

  std::vector<T> temp_values(count);
  for (SizeT i = 0; i != count; ++i)
    temp_values[i] = this->Values[sort_order[i]];
  std::swap(temp_values, this->Values);
}

template <typename T>
std::vector<typename svtkSparseArray<T>::CoordinateT> svtkSparseArray<T>::GetUniqueCoordinates(
  DimensionT dimension)
{
  if (dimension < 0 || dimension >= this->GetDimensions())
  {
    svtkErrorMacro(<< "Dimension out-of-bounds.");
    return std::vector<CoordinateT>();
  }

  std::vector<CoordinateT> results(this->Coordinates[dimension]);
  std::sort(results.begin(), results.end());
  results.erase(std::unique(results.begin(), results.end()), results.end());
  return results;
}

template <typename T>
const typename svtkSparseArray<T>::CoordinateT* svtkSparseArray<T>::GetCoordinateStorage(
  DimensionT dimension) const
{
  if (dimension < 0 || dimension >= this->GetDimensions())
  {
    svtkErrorMacro(<< "Dimension out-of-bounds.");
    return nullptr;
  }

  return &this->Coordinates[dimension][0];
}

template <typename T>
typename svtkSparseArray<T>::CoordinateT* svtkSparseArray<T>::GetCoordinateStorage(
  DimensionT dimension)
{
  if (dimension < 0 || dimension >= this->GetDimensions())
  {
    svtkErrorMacro(<< "Dimension out-of-bounds.");
    return nullptr;
  }

  return &this->Coordinates[dimension][0];
}

template <typename T>
const T* svtkSparseArray<T>::GetValueStorage() const
{
  return &(this->Values[0]);
}

template <typename T>
T* svtkSparseArray<T>::GetValueStorage()
{
  return &this->Values[0];
}

template <typename T>
void svtkSparseArray<T>::ReserveStorage(const SizeT value_count)
{
  for (DimensionT dimension = 0; dimension != this->GetDimensions(); ++dimension)
    this->Coordinates[dimension].resize(value_count);

  this->Values.resize(value_count);
}

template <typename T>
void svtkSparseArray<T>::SetExtentsFromContents()
{
  svtkArrayExtents new_extents;

  const svtkIdType row_begin = 0;
  const svtkIdType row_end = row_begin + static_cast<svtkIdType>(this->Values.size());
  const DimensionT dimension_count = this->GetDimensions();
  for (DimensionT dimension = 0; dimension != dimension_count; ++dimension)
  {
    svtkIdType range_begin = std::numeric_limits<svtkIdType>::max();
    svtkIdType range_end = -std::numeric_limits<svtkIdType>::max();
    for (svtkIdType row = row_begin; row != row_end; ++row)
    {
      range_begin = std::min(range_begin, this->Coordinates[dimension][row]);
      range_end = std::max(range_end, this->Coordinates[dimension][row] + 1);
    }
    new_extents.Append(svtkArrayRange(range_begin, range_end));
  }

  this->Extents = new_extents;
}

template <typename T>
void svtkSparseArray<T>::SetExtents(const svtkArrayExtents& extents)
{
  if (extents.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Extent-array dimension mismatch.");
    return;
  }

  this->Extents = extents;
}

template <typename T>
void svtkSparseArray<T>::AddValue(CoordinateT i, const T& value)
{
  this->AddValue(svtkArrayCoordinates(i), value);
}

template <typename T>
void svtkSparseArray<T>::AddValue(CoordinateT i, CoordinateT j, const T& value)
{
  this->AddValue(svtkArrayCoordinates(i, j), value);
}

template <typename T>
void svtkSparseArray<T>::AddValue(CoordinateT i, CoordinateT j, CoordinateT k, const T& value)
{
  this->AddValue(svtkArrayCoordinates(i, j, k), value);
}

template <typename T>
void svtkSparseArray<T>::AddValue(const svtkArrayCoordinates& coordinates, const T& value)
{
  if (coordinates.GetDimensions() != this->GetDimensions())
  {
    svtkErrorMacro(<< "Index-array dimension mismatch.");
    return;
  }

  this->Values.push_back(value);

  for (DimensionT i = 0; i != coordinates.GetDimensions(); ++i)
    this->Coordinates[i].push_back(coordinates[i]);
}

template <typename T>
bool svtkSparseArray<T>::Validate()
{
  svtkIdType duplicate_count = 0;
  svtkIdType out_of_bound_count = 0;

  const svtkIdType dimensions = this->GetDimensions();
  const svtkIdType count = this->GetNonNullSize();

  // Create an arbitrary sorted order for our coordinates ...
  svtkArraySort sort;
  sort.SetDimensions(dimensions);
  for (svtkIdType i = 0; i != dimensions; ++i)
    sort[i] = i;

  std::vector<svtkIdType> sort_order(count);
  for (svtkIdType i = 0; i != count; ++i)
    sort_order[i] = i;

  std::sort(sort_order.begin(), sort_order.end(), SortCoordinates(sort, this->Coordinates));

  // Now, look for duplicates ...
  for (svtkIdType i = 0; i + 1 < count; ++i)
  {
    svtkIdType j;
    for (j = 0; j != dimensions; ++j)
    {
      if (this->Coordinates[j][sort_order[i]] != this->Coordinates[j][sort_order[i + 1]])
        break;
    }
    if (j == dimensions)
    {
      duplicate_count += 1;
    }
  }

  // Look for out-of-bound coordinates ...
  for (svtkIdType i = 0; i != count; ++i)
  {
    for (svtkIdType j = 0; j != dimensions; ++j)
    {
      if (this->Coordinates[j][i] < this->Extents[j].GetBegin() ||
        this->Coordinates[j][i] >= this->Extents[j].GetEnd())
      {
        ++out_of_bound_count;
        break;
      }
    }
  }

  if (duplicate_count)
  {
    svtkErrorMacro(<< "Array contains " << duplicate_count << " duplicate coordinates.");
  }

  if (out_of_bound_count)
  {
    svtkErrorMacro(<< "Array contains " << out_of_bound_count << " out-of-bound coordinates.");
  }

  return (0 == duplicate_count) && (0 == out_of_bound_count);
}

template <typename T>
svtkSparseArray<T>::svtkSparseArray()
  : NullValue(T())
{
}

template <typename T>
svtkSparseArray<T>::~svtkSparseArray()
{
}

template <typename T>
void svtkSparseArray<T>::InternalResize(const svtkArrayExtents& extents)
{
  this->Extents = extents;
  this->DimensionLabels.resize(extents.GetDimensions(), svtkStdString());
  this->Coordinates.resize(extents.GetDimensions());
  this->Values.resize(0);
}

template <typename T>
void svtkSparseArray<T>::InternalSetDimensionLabel(DimensionT i, const svtkStdString& label)
{
  this->DimensionLabels[i] = label;
}

template <typename T>
svtkStdString svtkSparseArray<T>::InternalGetDimensionLabel(DimensionT i)
{
  return this->DimensionLabels[i];
}

#endif

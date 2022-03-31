/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericDataArrayLookupHelper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericDataArrayLookupHelper
 * @brief   internal class used by
 * svtkGenericDataArray to support LookupValue.
 *
 */

#ifndef svtkGenericDataArrayLookupHelper_h
#define svtkGenericDataArrayLookupHelper_h

#include "svtkIdList.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <limits>

namespace svtk
{
namespace detail
{
template <typename T, bool>
struct has_NaN;

template <typename T>
struct has_NaN<T, true>
{
  static bool isnan(T x) { return std::isnan(x); }
};

template <typename T>
struct has_NaN<T, false>
{
  static bool isnan(T) { return false; }
};

template <typename T>
bool isnan(T x)
{
  // Select the correct partially specialized type.
  return has_NaN<T, std::numeric_limits<T>::has_quiet_NaN>::isnan(x);
}
} // namespace detail
}

template <class ArrayTypeT>
class svtkGenericDataArrayLookupHelper
{
public:
  typedef ArrayTypeT ArrayType;
  typedef typename ArrayType::ValueType ValueType;

  svtkGenericDataArrayLookupHelper() = default;

  ~svtkGenericDataArrayLookupHelper() { this->ClearLookup(); }

  void SetArray(ArrayTypeT* array)
  {
    if (this->AssociatedArray != array)
    {
      this->ClearLookup();
      this->AssociatedArray = array;
    }
  }

  svtkIdType LookupValue(ValueType elem)
  {
    this->UpdateLookup();
    auto indices = FindIndexVec(elem);
    if (indices == nullptr)
    {
      return -1;
    }
    return indices->front();
  }

  void LookupValue(ValueType elem, svtkIdList* ids)
  {
    ids->Reset();
    this->UpdateLookup();
    auto indices = FindIndexVec(elem);
    if (indices)
    {
      ids->Allocate(static_cast<svtkIdType>(indices->size()));
      for (auto index : *indices)
      {
        ids->InsertNextId(index);
      }
    }
  }

  //@{
  /**
   * Release any allocated memory for internal data-structures.
   */
  void ClearLookup()
  {
    this->ValueMap.clear();
    this->NanIndices.clear();
  }
  //@}

private:
  svtkGenericDataArrayLookupHelper(const svtkGenericDataArrayLookupHelper&) = delete;
  void operator=(const svtkGenericDataArrayLookupHelper&) = delete;

  void UpdateLookup()
  {
    if (!this->AssociatedArray || (this->AssociatedArray->GetNumberOfTuples() < 1) ||
      (!this->ValueMap.empty() || !this->NanIndices.empty()))
    {
      return;
    }

    svtkIdType num = this->AssociatedArray->GetNumberOfValues();
    this->ValueMap.reserve(num);
    for (svtkIdType i = 0; i < num; ++i)
    {
      auto value = this->AssociatedArray->GetValue(i);
      if (svtk::detail::isnan(value))
      {
        NanIndices.push_back(i);
      }
      this->ValueMap[value].push_back(i);
    }
  }

  // Return a pointer to the relevant vector of indices if specified value was
  // found in the array.
  std::vector<svtkIdType>* FindIndexVec(ValueType value)
  {
    std::vector<svtkIdType>* indices{ nullptr };
    if (svtk::detail::isnan(value) && !this->NanIndices.empty())
    {
      indices = &this->NanIndices;
    }
    const auto& pos = this->ValueMap.find(value);
    if (pos != this->ValueMap.end())
    {
      indices = &pos->second;
    }
    return indices;
  }

  ArrayTypeT* AssociatedArray{ nullptr };
  std::unordered_map<ValueType, std::vector<svtkIdType> > ValueMap;
  std::vector<svtkIdType> NanIndices;
};

#endif
// SVTK-HeaderTest-Exclude: svtkGenericDataArrayLookupHelper.h

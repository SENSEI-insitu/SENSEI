/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayValueRange_AOS.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * Specialization of value ranges and iterators for svtkAOSDataArrayTemplate.
 */

#ifndef svtkDataArrayValueRange_AOS_h
#define svtkDataArrayValueRange_AOS_h

#include "svtkAOSDataArrayTemplate.h"
#include "svtkDataArrayMeta.h"
#include "svtkDataArrayValueRange_Generic.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <type_traits>

#ifndef __SVTK_WRAP__

// Disable this specialization when iterator debugging is requested:
#ifndef SVTK_DEBUG_RANGE_ITERATORS

SVTK_ITER_OPTIMIZE_START

namespace svtk
{

namespace detail
{

//------------------------------------------------------------------------------
// ValueRange
template <typename ValueTypeT, ComponentIdType TupleSize>
struct ValueRange<svtkAOSDataArrayTemplate<ValueTypeT>, TupleSize>
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");

  using IdStorageType = IdStorage<TupleSize>;
  using NumCompsType = GenericTupleSize<TupleSize>;

public:
  using ArrayType = svtkAOSDataArrayTemplate<ValueTypeT>;
  using ValueType = ValueTypeT;

  using IteratorType = ValueType*;
  using ConstIteratorType = ValueType const*;
  using ReferenceType = ValueType&;
  using ConstReferenceType = ValueType const&;

  // May be DynamicTupleSize, or the actual tuple size.
  constexpr static ComponentIdType TupleSizeTag = TupleSize;

  // STL-compat
  using value_type = ValueType;
  using size_type = ValueIdType;
  using iterator = IteratorType;
  using const_iterator = ConstIteratorType;
  using reference = ReferenceType;
  using const_reference = ConstReferenceType;

  SVTK_ITER_INLINE
  ValueRange() noexcept = default;

  SVTK_ITER_INLINE
  ValueRange(ArrayType* arr, ValueIdType beginValue, ValueIdType endValue) noexcept
    : Array(arr)
    , NumComps(arr)
    , Begin(arr->GetPointer(beginValue))
    , End(arr->GetPointer(endValue))
  {
    assert(this->Array);
    assert(beginValue >= 0 && beginValue <= endValue);
    assert(endValue >= 0 && endValue <= this->Array->GetNumberOfValues());
  }

  SVTK_ITER_INLINE
  ValueRange GetSubRange(ValueIdType beginValue = 0, ValueIdType endValue = -1) const noexcept
  {
    const ValueIdType realBegin =
      std::distance(this->Array->GetPointer(0), this->Begin) + beginValue;
    const ValueIdType realEnd = endValue >= 0
      ? std::distance(this->Array->GetPointer(0), this->Begin) + endValue
      : std::distance(this->Array->GetPointer(0), this->End);

    return ValueRange{ this->Array, realBegin, realEnd };
  }

  SVTK_ITER_INLINE
  ArrayType* GetArray() const noexcept { return this->Array; }

  SVTK_ITER_INLINE
  ComponentIdType GetTupleSize() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  ValueIdType GetBeginValueId() const noexcept
  {
    return static_cast<ValueIdType>(this->Begin - this->Array->GetPointer(0));
  }

  SVTK_ITER_INLINE
  ValueIdType GetEndValueId() const noexcept
  {
    return static_cast<ValueIdType>(this->End - this->Array->GetPointer(0));
  }

  SVTK_ITER_INLINE
  size_type size() const noexcept { return static_cast<size_type>(this->End - this->Begin); }

  SVTK_ITER_INLINE
  iterator begin() noexcept { return this->Begin; }
  SVTK_ITER_INLINE
  iterator end() noexcept { return this->End; }

  SVTK_ITER_INLINE
  const_iterator begin() const noexcept { return this->Begin; }
  SVTK_ITER_INLINE
  const_iterator end() const noexcept { return this->End; }

  SVTK_ITER_INLINE
  const_iterator cbegin() const noexcept { return this->Begin; }
  SVTK_ITER_INLINE
  const_iterator cend() const noexcept { return this->End; }

  SVTK_ITER_INLINE
  reference operator[](size_type i) noexcept { return this->Begin[i]; }
  SVTK_ITER_INLINE
  const_reference operator[](size_type i) const noexcept { return this->Begin[i]; }

private:
  mutable ArrayType* Array{ nullptr };
  NumCompsType NumComps{};
  ValueType* Begin{ nullptr };
  ValueType* End{ nullptr };
};

// Unimplemented, only used inside decltype in SelectValueRange:
template <typename ArrayType, ComponentIdType TupleSize,
  // Convenience:
  typename ValueType = typename ArrayType::ValueType,
  typename AOSArrayType = svtkAOSDataArrayTemplate<ValueType>,
  // SFINAE to select AOS arrays:
  typename = typename std::enable_if<IsAOSDataArray<ArrayType>::value>::type>
ValueRange<AOSArrayType, TupleSize> DeclareValueRangeSpecialization(ArrayType*);

}
} // end namespace svtk::detail

SVTK_ITER_OPTIMIZE_END

#endif // SVTK_DEBUG_RANGE_ITERATORS
#endif // __SVTK_WRAP__
#endif // svtkDataArrayValueRange_AOS_h

// SVTK-HeaderTest-Exclude: svtkDataArrayValueRange_AOS.h

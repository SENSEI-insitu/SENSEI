/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayTupleRange_AOS.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * Specialization of tuple ranges and iterators for svtkAOSDataArrayTemplate.
 */

#ifndef svtkDataArrayTupleRange_AOS_h
#define svtkDataArrayTupleRange_AOS_h

#include "svtkAOSDataArrayTemplate.h"
#include "svtkDataArrayMeta.h"
#include "svtkDataArrayTupleRange_Generic.h"
#include "svtkIterator.h"

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

// Forward decs for friends/args
template <typename ArrayType, ComponentIdType>
struct ConstTupleReference;
template <typename ArrayType, ComponentIdType>
struct TupleReference;
template <typename ArrayType, ComponentIdType>
struct ConstTupleIterator;
template <typename ArrayType, ComponentIdType>
struct TupleIterator;
template <typename ArrayType, ComponentIdType>
struct TupleRange;

//------------------------------------------------------------------------------
// Const tuple reference
template <typename ValueType, ComponentIdType TupleSize>
struct ConstTupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>
{
private:
  using ArrayType = svtkAOSDataArrayTemplate<ValueType>;
  using NumCompsType = GenericTupleSize<TupleSize>;
  using APIType = ValueType;

public:
  using size_type = ComponentIdType;
  using value_type = APIType;
  using const_reference = const ValueType&;
  using iterator = const ValueType*;
  using const_iterator = const ValueType*;

  SVTK_ITER_INLINE
  ConstTupleReference() noexcept : Tuple{ nullptr } {}

  SVTK_ITER_INLINE
  ConstTupleReference(const ValueType* tuple, NumCompsType numComps) noexcept
    : Tuple(tuple)
    , NumComps(numComps)
  {
  }

  SVTK_ITER_INLINE
  ConstTupleReference(const TupleReference<ArrayType, TupleSize>& o) noexcept
    : Tuple{ o.Tuple }
    , NumComps{ o.NumComps }
  {
  }

  SVTK_ITER_INLINE
  ConstTupleReference(const ConstTupleReference&) noexcept = default;
  SVTK_ITER_INLINE
  ConstTupleReference(ConstTupleReference&&) noexcept = default;

  // Allow this type to masquerade as a pointer, so that tupleIiter->foo works.
  SVTK_ITER_INLINE
  ConstTupleReference* operator->() noexcept { return this; }
  SVTK_ITER_INLINE
  const ConstTupleReference* operator->() const noexcept { return this; }

  // Caller must ensure that there are size() elements in array.
  SVTK_ITER_INLINE void GetTuple(volatile APIType* tuple) const noexcept
  {
    // Yes, the tuple argument is marked volatile. No, it's not a mistake.
    //
    // `volatile`'s intended usage per the standard is to disable optimizations
    // when accessing a variable. Without it, GCC 8 will optimize the following
    // loop to memcpy, but we're usually copying small tuples here, and the
    // call to memcpy is more expensive than just doing an inline copy. By
    // disabling the memcpy optimization, benchmarks are 60% faster when
    // iterating with the Get/SetTuple methods, and are comparable to other
    // methods of array access.
    SVTK_ITER_ASSUME(this->NumComps.value > 0);
    for (ComponentIdType i = 0; i < this->NumComps.value; ++i)
    {
      tuple[i] = this->Tuple[i];
    }
  }

  // skips some runtime checks when both sizes are fixed:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, bool> operator==(
    const TupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // Needs a runtime check:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, bool> operator==(
    const TupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Need to check the size at runtime :-(
    if (other.size() != this->NumComps.value)
    {
      return false;
    }

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // skips some runtime checks when both sizes are fixed:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, bool> operator==(
    const ConstTupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // Needs a runtime check:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, bool> operator==(
    const ConstTupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Need to check the size at runtime :-(
    if (other.size() != this->NumComps.value)
    {
      return false;
    }

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE bool operator!=(const TupleReference<OArrayType, OSize>& o) const noexcept
  {
    return !(*this == o);
  }

  template <typename OArray, ComponentIdType OSize>
  SVTK_ITER_INLINE bool operator!=(const ConstTupleReference<OArray, OSize>& o) const noexcept
  {
    return !(*this == o);
  }

  SVTK_ITER_INLINE
  const_reference operator[](size_type i) const noexcept { return this->Tuple[i]; }

  SVTK_ITER_INLINE
  size_type size() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  const_iterator begin() const noexcept { return const_iterator{ this->Tuple }; }

  SVTK_ITER_INLINE
  const_iterator end() const noexcept
  {
    return const_iterator{ this->Tuple + this->NumComps.value };
  }

  SVTK_ITER_INLINE
  const_iterator cbegin() const noexcept { return const_iterator{ this->Tuple }; }

  SVTK_ITER_INLINE
  const_iterator cend() const noexcept
  {
    return const_iterator{ this->Tuple + this->NumComps.value };
  }

  friend struct ConstTupleIterator<ArrayType, TupleSize>;

protected:
  // Intentionally hidden:
  SVTK_ITER_INLINE
  ConstTupleReference& operator=(const ConstTupleReference&) noexcept = default;

  const ValueType* Tuple;
  NumCompsType NumComps;
};

//------------------------------------------------------------------------------
// Tuple reference
template <typename ValueType, ComponentIdType TupleSize>
struct TupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>
{
private:
  using ArrayType = svtkAOSDataArrayTemplate<ValueType>;
  using NumCompsType = GenericTupleSize<TupleSize>;
  using APIType = ValueType;

public:
  using size_type = ComponentIdType;
  using value_type = APIType;
  using iterator = ValueType*;
  using const_iterator = const ValueType*;
  using reference = ValueType&;
  using const_reference = ValueType const&;

  SVTK_ITER_INLINE
  TupleReference() noexcept : Tuple{ nullptr } {}

  SVTK_ITER_INLINE
  TupleReference(ValueType* tuple, NumCompsType numComps) noexcept
    : Tuple(tuple)
    , NumComps(numComps)
  {
  }

  SVTK_ITER_INLINE
  TupleReference(const TupleReference&) noexcept = default;
  SVTK_ITER_INLINE
  TupleReference(TupleReference&&) noexcept = default;

  // Allow this type to masquerade as a pointer, so that tupleIiter->foo works.
  SVTK_ITER_INLINE
  TupleReference* operator->() noexcept { return this; }
  SVTK_ITER_INLINE
  const TupleReference* operator->() const noexcept { return this; }

  // Caller must ensure that there are size() elements in array.
  SVTK_ITER_INLINE
  void GetTuple(volatile APIType* tuple) const noexcept
  {
    // Yes, the tuple argument is marked volatile. No, it's not a mistake.
    //
    // `volatile`'s intended usage per the standard is to disable optimizations
    // when accessing a variable. Without it, GCC 8 will optimize the following
    // loop to memcpy, but we're usually copying small tuples here, and the
    // call to memcpy is more expensive than just doing an inline copy. By
    // disabling the memcpy optimization, benchmarks are 60% faster when
    // iterating with the Get/SetTuple methods, and are comparable to other
    // methods of array access.
    SVTK_ITER_ASSUME(this->NumComps.value > 0);
    for (ComponentIdType i = 0; i < this->NumComps.value; ++i)
    {
      tuple[i] = this->Tuple[i];
    }
  }

  // Caller must ensure that there are size() elements in array.
  SVTK_ITER_INLINE
  void SetTuple(const APIType* tuple) noexcept
  {
    volatile APIType* out = this->Tuple;
    // Yes, this variable argument is marked volatile. See the explanation in
    // GetTuple.
    SVTK_ITER_ASSUME(this->NumComps.value > 0);
    for (ComponentIdType i = 0; i < this->NumComps.value; ++i)
    {
      out[i] = tuple[i];
    }
  }

  SVTK_ITER_INLINE
  TupleReference& operator=(const TupleReference& other) noexcept
  {
    std::copy_n(other.cbegin(), this->NumComps.value, this->begin());
    return *this;
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, TupleReference&> operator=(
    const TupleReference<OArrayType, OSize>& other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    std::copy_n(other.cbegin(), OSize, this->begin());
    return *this;
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, TupleReference&> operator=(
    const TupleReference<OArrayType, OSize>& other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Note that the sizes are not checked here. Enable
    // SVTK_DEBUG_RANGE_ITERATORS to enable check.
    std::copy_n(other.cbegin(), this->NumComps.value, this->begin());
    return *this;
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, TupleReference&> operator=(
    const ConstTupleReference<OArrayType, OSize>& other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    std::copy_n(other.cbegin(), OSize, this->begin());
    return *this;
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, TupleReference&> operator=(
    const ConstTupleReference<OArrayType, OSize>& other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Note that the sizes are not checked here. Enable
    // SVTK_DEBUG_RANGE_ITERATORS to enable check.
    std::copy_n(other.cbegin(), this->NumComps.value, this->begin());
    return *this;
  }

  // skips some runtime checks when both sizes are fixed:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, bool> operator==(
    const TupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // Needs a runtime check:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, bool> operator==(
    const TupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Note that the sizes are not checked here. Enable
    // SVTK_DEBUG_RANGE_ITERATORS to enable check.
    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // skips some runtime checks when both sizes are fixed:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, bool> operator==(
    const ConstTupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot assign tuples with different sizes.");

    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  // Needs a runtime check:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, bool> operator==(
    const ConstTupleReference<OArrayType, OSize>& other) const noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert(
      (std::is_convertible<OAPIType, APIType>{}), "Incompatible types when assigning tuples.");

    // Note that the sizes are not checked here. Enable
    // SVTK_DEBUG_RANGE_ITERATORS to enable check.
    return std::equal(this->cbegin(), this->cend(), other.cbegin());
  }

  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE bool operator!=(const TupleReference<OArrayType, OSize>& o) const noexcept
  {
    return !(*this == o);
  }

  template <typename OArray, ComponentIdType OSize>
  SVTK_ITER_INLINE bool operator!=(const ConstTupleReference<OArray, OSize>& o) const noexcept
  {
    return !(*this == o);
  }

  // skips some runtime checks:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfStaticTupleSizes<TupleSize, OSize, void> swap(
    TupleReference<OArrayType, OSize> other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert((std::is_same<OAPIType, APIType>{}), "Incompatible types when swapping tuples.");

    // SFINAE guarantees that the tuple sizes are not dynamic in this overload:
    static_assert(TupleSize == OSize, "Cannot swap tuples with different sizes.");

    std::swap_ranges(this->begin(), this->end(), other.begin());
  }

  // Needs a runtime check:
  template <typename OArrayType, ComponentIdType OSize>
  SVTK_ITER_INLINE EnableIfEitherTupleSizeIsDynamic<TupleSize, OSize, void> swap(
    TupleReference<OArrayType, OSize> other) noexcept
  {
    // Check that types are convertible:
    using OAPIType = GetAPIType<OArrayType>;
    static_assert((std::is_same<OAPIType, APIType>{}), "Incompatible types when swapping tuples.");

    // Note that the sizes are not checked here. Enable
    // SVTK_DEBUG_RANGE_ITERATORS to enable check.
    std::swap_ranges(this->begin(), this->end(), other.begin());
  }

  friend SVTK_ITER_INLINE void swap(TupleReference a, TupleReference b) noexcept { a.swap(b); }

  template <typename OArray, ComponentIdType OSize>
  friend SVTK_ITER_INLINE void swap(TupleReference a, TupleReference<OArray, OSize> b) noexcept
  {
    a.swap(b);
  }

  SVTK_ITER_INLINE
  reference operator[](size_type i) noexcept { return this->Tuple[i]; }

  SVTK_ITER_INLINE
  const_reference operator[](size_type i) const noexcept { return this->Tuple[i]; }

  SVTK_ITER_INLINE
  void fill(const value_type& v) noexcept { std::fill(this->begin(), this->end(), v); }

  SVTK_ITER_INLINE
  size_type size() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  iterator begin() noexcept { return iterator{ this->Tuple }; }

  SVTK_ITER_INLINE
  iterator end() noexcept { return iterator{ this->Tuple + this->NumComps.value }; }

  SVTK_ITER_INLINE
  const_iterator begin() const noexcept { return const_iterator{ this->Tuple }; }

  SVTK_ITER_INLINE
  const_iterator end() const noexcept
  {
    return const_iterator{ this->Tuple + this->NumComps.value };
  }

  SVTK_ITER_INLINE
  const_iterator cbegin() const noexcept { return const_iterator{ this->Tuple }; }

  SVTK_ITER_INLINE
  const_iterator cend() const noexcept
  {
    return const_iterator{ this->Tuple + this->NumComps.value };
  }

  friend struct ConstTupleReference<ArrayType, TupleSize>;
  friend struct TupleIterator<ArrayType, TupleSize>;

protected:
  SVTK_ITER_INLINE
  void CopyReference(const TupleReference& o) noexcept
  {
    this->Tuple = o.Tuple;
    this->NumComps = o.NumComps;
  }

  ValueType* Tuple;
  NumCompsType NumComps;
};

//------------------------------------------------------------------------------
// Const tuple iterator
template <typename ValueType, ComponentIdType TupleSize>
struct ConstTupleIterator<svtkAOSDataArrayTemplate<ValueType>, TupleSize>
  : public svtkIterator<std::random_access_iterator_tag,
      ConstTupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>, TupleIdType,
      ConstTupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>,
      ConstTupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize> >
{
private:
  using ArrayType = svtkAOSDataArrayTemplate<ValueType>;
  using NumCompsType = GenericTupleSize<TupleSize>;
  using Superclass = svtkIterator<std::random_access_iterator_tag,
    ConstTupleReference<ArrayType, TupleSize>, TupleIdType,
    ConstTupleReference<ArrayType, TupleSize>, ConstTupleReference<ArrayType, TupleSize> >;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  SVTK_ITER_INLINE
  ConstTupleIterator() noexcept = default;

  SVTK_ITER_INLINE
  ConstTupleIterator(const ValueType* tuple, NumCompsType numComps) noexcept : Ref(tuple, numComps)
  {
  }

  SVTK_ITER_INLINE
  ConstTupleIterator(const TupleIterator<ArrayType, TupleSize>& o) noexcept : Ref{ o.Ref } {}

  SVTK_ITER_INLINE
  ConstTupleIterator(const ConstTupleIterator& o) noexcept = default;
  SVTK_ITER_INLINE
  ConstTupleIterator& operator=(const ConstTupleIterator& o) noexcept = default;

  SVTK_ITER_INLINE
  ConstTupleIterator& operator++() noexcept // prefix
  {
    this->Ref.Tuple += this->Ref.NumComps.value;
    return *this;
  }

  SVTK_ITER_INLINE
  ConstTupleIterator operator++(int) noexcept // postfix
  {
    auto tuple = this->Ref.Tuple;
    this->Ref.Tuple += this->Ref.NumComps.value;
    return ConstTupleIterator{ tuple, this->Ref.NumComps };
  }

  SVTK_ITER_INLINE
  ConstTupleIterator& operator--() noexcept // prefix
  {
    this->Ref.Tuple -= this->Ref.NumComps.value;
    return *this;
  }

  SVTK_ITER_INLINE
  ConstTupleIterator operator--(int) noexcept // postfix
  {
    auto tuple = this->Ref.Tuple;
    this->Ref.Tuple -= this->Ref.NumComps.value;
    return ConstTupleIterator{ tuple, this->Ref.NumComps };
  }

  SVTK_ITER_INLINE
  reference operator[](difference_type i) noexcept
  {
    return reference{ this->Ref.Tuple + i * this->Ref.NumComps, this->Ref.NumComps };
  }

  SVTK_ITER_INLINE
  reference operator*() noexcept { return this->Ref; }

  SVTK_ITER_INLINE
  pointer& operator->() noexcept { return this->Ref; }

#define SVTK_TMP_MAKE_OPERATOR(OP)                                                                  \
  friend SVTK_ITER_INLINE bool operator OP(                                                         \
    const ConstTupleIterator& lhs, const ConstTupleIterator& rhs) noexcept                         \
  {                                                                                                \
    return lhs.GetTuple() OP rhs.GetTuple();                                                       \
  }

  SVTK_TMP_MAKE_OPERATOR(==)
  SVTK_TMP_MAKE_OPERATOR(!=)
  SVTK_TMP_MAKE_OPERATOR(<)
  SVTK_TMP_MAKE_OPERATOR(>)
  SVTK_TMP_MAKE_OPERATOR(<=)
  SVTK_TMP_MAKE_OPERATOR(>=)

#undef SVTK_TMP_MAKE_OPERATOR

  SVTK_ITER_INLINE
  ConstTupleIterator& operator+=(difference_type offset) noexcept
  {
    this->Ref.Tuple += offset * this->Ref.NumComps.value;
    return *this;
  }

  friend SVTK_ITER_INLINE ConstTupleIterator operator+(
    const ConstTupleIterator& it, difference_type offset) noexcept
  {
    return ConstTupleIterator{ it.GetTuple() + offset * it.GetNumComps().value, it.GetNumComps() };
  }

  friend SVTK_ITER_INLINE ConstTupleIterator operator+(
    difference_type offset, const ConstTupleIterator& it) noexcept
  {
    return ConstTupleIterator{ it.GetTuple() + offset * it.GetNumComps().value, it.GetNumComps() };
  }

  SVTK_ITER_INLINE
  ConstTupleIterator& operator-=(difference_type offset) noexcept
  {
    this->Ref.Tuple -= offset * this->Ref.NumComps.value;
    return *this;
  }

  friend SVTK_ITER_INLINE ConstTupleIterator operator-(
    const ConstTupleIterator& it, difference_type offset) noexcept
  {
    return ConstTupleIterator{ it.GetTuple() - offset * it.GetNumComps().value, it.GetNumComps() };
  }

  friend SVTK_ITER_INLINE difference_type operator-(
    const ConstTupleIterator& it1, const ConstTupleIterator& it2) noexcept
  {
    return static_cast<difference_type>(
      (it1.GetTuple() - it2.GetTuple()) / it1.GetNumComps().value);
  }

  friend SVTK_ITER_INLINE void swap(ConstTupleIterator& lhs, ConstTupleIterator& rhs) noexcept
  {
    using std::swap;
    swap(lhs.GetTuple(), rhs.GetTuple());
    swap(lhs.GetNumComps(), rhs.GetNumComps());
  }

private:
  SVTK_ITER_INLINE
  const ValueType*& GetTuple() noexcept { return this->Ref.Tuple; }
  SVTK_ITER_INLINE
  const ValueType* GetTuple() const noexcept { return this->Ref.Tuple; }
  SVTK_ITER_INLINE
  NumCompsType& GetNumComps() noexcept { return this->Ref.NumComps; }
  SVTK_ITER_INLINE
  NumCompsType GetNumComps() const noexcept { return this->Ref.NumComps; }

  ConstTupleReference<ArrayType, TupleSize> Ref;
};

//------------------------------------------------------------------------------
// Tuple iterator
template <typename ValueType, ComponentIdType TupleSize>
struct TupleIterator<svtkAOSDataArrayTemplate<ValueType>, TupleSize>
  : public svtkIterator<std::random_access_iterator_tag,
      TupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>, TupleIdType,
      TupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize>,
      TupleReference<svtkAOSDataArrayTemplate<ValueType>, TupleSize> >
{
private:
  using ArrayType = svtkAOSDataArrayTemplate<ValueType>;
  using NumCompsType = GenericTupleSize<TupleSize>;
  using Superclass =
    svtkIterator<std::random_access_iterator_tag, TupleReference<ArrayType, TupleSize>,
      TupleIdType, TupleReference<ArrayType, TupleSize>, TupleReference<ArrayType, TupleSize> >;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  SVTK_ITER_INLINE
  TupleIterator() noexcept = default;

  SVTK_ITER_INLINE
  TupleIterator(ValueType* tuple, NumCompsType numComps) noexcept : Ref(tuple, numComps) {}

  SVTK_ITER_INLINE
  TupleIterator(const TupleIterator& o) noexcept = default;

  SVTK_ITER_INLINE
  TupleIterator& operator=(const TupleIterator& o) noexcept
  {
    this->Ref.CopyReference(o.Ref);
    return *this;
  }

  SVTK_ITER_INLINE
  TupleIterator& operator++() noexcept // prefix
  {
    this->Ref.Tuple += this->Ref.NumComps.value;
    return *this;
  }

  SVTK_ITER_INLINE
  TupleIterator operator++(int) noexcept // postfix
  {
    auto tuple = this->Ref.Tuple;
    this->Ref.Tuple += this->Ref.NumComps.value;
    return TupleIterator{ tuple, this->Ref.NumComps };
  }

  SVTK_ITER_INLINE
  TupleIterator& operator--() noexcept // prefix
  {
    this->Ref.Tuple -= this->Ref.NumComps.value;
    return *this;
  }

  SVTK_ITER_INLINE
  TupleIterator operator--(int) noexcept // postfix
  {
    auto tuple = this->Ref.Tuple;
    this->Ref.Tuple -= this->Ref.NumComps.value;
    return TupleIterator{ tuple, this->Ref.NumComps };
  }

  SVTK_ITER_INLINE
  reference operator[](difference_type i) noexcept
  {
    return reference{ this->Ref.Tuple + i * this->Ref.NumComps.value, this->Ref.NumComps };
  }

  reference operator*() noexcept { return this->Ref; }

  pointer& operator->() noexcept { return this->Ref; }

#define SVTK_TMP_MAKE_OPERATOR(OP)                                                                  \
  friend SVTK_ITER_INLINE bool operator OP(const TupleIterator& lhs, const TupleIterator& rhs)      \
    noexcept                                                                                       \
  {                                                                                                \
    return lhs.GetTuple() OP rhs.GetTuple();                                                       \
  }

  SVTK_TMP_MAKE_OPERATOR(==)
  SVTK_TMP_MAKE_OPERATOR(!=)
  SVTK_TMP_MAKE_OPERATOR(<)
  SVTK_TMP_MAKE_OPERATOR(>)
  SVTK_TMP_MAKE_OPERATOR(<=)
  SVTK_TMP_MAKE_OPERATOR(>=)

#undef SVTK_TMP_MAKE_OPERATOR

  SVTK_ITER_INLINE
  TupleIterator& operator+=(difference_type offset) noexcept
  {
    this->Ref.Tuple += offset * this->Ref.NumComps.value;
    return *this;
  }

  friend SVTK_ITER_INLINE TupleIterator operator+(
    const TupleIterator& it, difference_type offset) noexcept
  {
    return TupleIterator{ it.GetTuple() + offset * it.GetNumComps().value, it.GetNumComps() };
  }

  friend SVTK_ITER_INLINE TupleIterator operator+(
    difference_type offset, const TupleIterator& it) noexcept
  {
    return TupleIterator{ it.GetTuple() + offset * it.GetNumComps().value, it.GetNumComps() };
  }

  SVTK_ITER_INLINE
  TupleIterator& operator-=(difference_type offset) noexcept
  {
    this->Ref.Tuple -= offset * this->Ref.NumComps.value;
    return *this;
  }

  friend SVTK_ITER_INLINE TupleIterator operator-(
    const TupleIterator& it, difference_type offset) noexcept
  {
    return TupleIterator{ it.GetTuple() - offset * it.GetNumComps().value, it.GetNumComps() };
  }

  friend SVTK_ITER_INLINE difference_type operator-(
    const TupleIterator& it1, const TupleIterator& it2) noexcept
  {
    return static_cast<difference_type>(
      (it1.GetTuple() - it2.GetTuple()) / it1.GetNumComps().value);
  }

  friend SVTK_ITER_INLINE void swap(TupleIterator& lhs, TupleIterator& rhs) noexcept
  {
    using std::swap;
    swap(lhs.GetTuple(), rhs.GetTuple());
    swap(lhs.GetNumComps(), rhs.GetNumComps());
  }

  friend struct ConstTupleIterator<ArrayType, TupleSize>;

protected:
  SVTK_ITER_INLINE
  ValueType* GetTuple() const noexcept { return this->Ref.Tuple; }
  SVTK_ITER_INLINE
  ValueType*& GetTuple() noexcept { return this->Ref.Tuple; }
  SVTK_ITER_INLINE
  NumCompsType GetNumComps() const noexcept { return this->Ref.NumComps; }
  SVTK_ITER_INLINE
  NumCompsType& GetNumComps() noexcept { return this->Ref.NumComps; }

  TupleReference<ArrayType, TupleSize> Ref;
};

//------------------------------------------------------------------------------
// Tuple range
template <typename ValueType, ComponentIdType TupleSize>
struct TupleRange<svtkAOSDataArrayTemplate<ValueType>, TupleSize>
{
  using ArrayType = svtkAOSDataArrayTemplate<ValueType>;
  using APIType = GetAPIType<ArrayType>;

private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayType>::value, "Invalid array type.");

  using NumCompsType = GenericTupleSize<TupleSize>;

public:
  using TupleIteratorType = TupleIterator<ArrayType, TupleSize>;
  using ConstTupleIteratorType = ConstTupleIterator<ArrayType, TupleSize>;
  using TupleReferenceType = TupleReference<ArrayType, TupleSize>;
  using ConstTupleReferenceType = ConstTupleReference<ArrayType, TupleSize>;
  using ComponentIteratorType = APIType*;
  using ConstComponentIteratorType = APIType const*;
  using ComponentReferenceType = APIType&;
  using ConstComponentReferenceType = const APIType&;
  using ComponentType = APIType;

  using size_type = TupleIdType;
  using iterator = TupleIteratorType;
  using const_iterator = ConstTupleIteratorType;
  using reference = TupleReferenceType;
  using const_reference = ConstTupleReferenceType;

  // May be DynamicTupleSize, or the actual tuple size.
  constexpr static ComponentIdType TupleSizeTag = TupleSize;

  SVTK_ITER_INLINE
  TupleRange() noexcept = default;

  SVTK_ITER_INLINE
  TupleRange(ArrayType* arr, TupleIdType beginTuple, TupleIdType endTuple) noexcept
    : Array(arr)
    , NumComps(arr)
    , BeginTuple(TupleRange::GetTuplePointer(arr, beginTuple))
    , EndTuple(TupleRange::GetTuplePointer(arr, endTuple))
  {
    assert(this->Array);
    assert(beginTuple >= 0 && beginTuple <= endTuple);
    assert(endTuple >= 0 && endTuple <= this->Array->GetNumberOfTuples());
  }

  SVTK_ITER_INLINE
  TupleRange GetSubRange(TupleIdType beginTuple = 0, TupleIdType endTuple = -1) const noexcept
  {
    const TupleIdType curBegin = this->GetTupleId(this->BeginTuple);
    const TupleIdType realBegin = curBegin + beginTuple;
    const TupleIdType realEnd =
      endTuple >= 0 ? curBegin + endTuple : this->GetTupleId(this->EndTuple);

    return TupleRange{ this->Array, realBegin, realEnd };
  }

  SVTK_ITER_INLINE
  ArrayType* GetArray() const noexcept { return this->Array; }

  SVTK_ITER_INLINE
  ComponentIdType GetTupleSize() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  TupleIdType GetBeginTupleId() const noexcept { return this->GetTupleId(this->BeginTuple); }

  SVTK_ITER_INLINE
  TupleIdType GetEndTupleId() const noexcept { return this->GetTupleId(this->EndTuple); }

  SVTK_ITER_INLINE
  size_type size() const noexcept
  {
    return static_cast<size_type>(this->EndTuple - this->BeginTuple) /
      static_cast<size_type>(this->NumComps.value);
  }

  SVTK_ITER_INLINE
  iterator begin() noexcept { return iterator(this->BeginTuple, this->NumComps); }

  SVTK_ITER_INLINE
  iterator end() noexcept { return iterator(this->EndTuple, this->NumComps); }

  SVTK_ITER_INLINE
  const_iterator begin() const noexcept { return const_iterator(this->BeginTuple, this->NumComps); }

  SVTK_ITER_INLINE
  const_iterator end() const noexcept { return const_iterator(this->EndTuple, this->NumComps); }

  SVTK_ITER_INLINE
  const_iterator cbegin() const noexcept
  {
    return const_iterator(this->BeginTuple, this->NumComps);
  }

  SVTK_ITER_INLINE
  const_iterator cend() const noexcept { return const_iterator(this->EndTuple, this->NumComps); }

  SVTK_ITER_INLINE
  reference operator[](size_type i) noexcept
  {
    return reference{ this->BeginTuple + i * this->NumComps.value, this->NumComps };
  }

  SVTK_ITER_INLINE
  const_reference operator[](size_type i) const noexcept
  {
    return const_reference{ this->BeginTuple + i * this->NumComps.value, this->NumComps };
  }

private:
  SVTK_ITER_INLINE
  ValueType* GetTuplePointer(ArrayType* array, svtkIdType tuple) const noexcept
  {
    return array->GetPointer(tuple * this->NumComps.value);
  }

  SVTK_ITER_INLINE
  TupleIdType GetTupleId(const ValueType* ptr) const noexcept
  {
    return static_cast<TupleIdType>((ptr - this->Array->GetPointer(0)) / this->NumComps.value);
  }

  mutable ArrayType* Array{ nullptr };
  NumCompsType NumComps{};
  ValueType* BeginTuple{ nullptr };
  ValueType* EndTuple{ nullptr };
};

// Unimplemented, only used inside decltype in SelectTupleRange:
template <typename ArrayType, ComponentIdType TupleSize,
  // Convenience:
  typename ValueType = typename ArrayType::ValueType,
  typename AOSArrayType = svtkAOSDataArrayTemplate<ValueType>,
  // SFINAE to select AOS arrays:
  typename = typename std::enable_if<IsAOSDataArray<ArrayType>::value>::type>
TupleRange<AOSArrayType, TupleSize> DeclareTupleRangeSpecialization(ArrayType*);

} // end namespace detail
} // end namespace svtk

SVTK_ITER_OPTIMIZE_END

#endif // SVTK_DEBUG_RANGE_ITERATORS
#endif // __SVTK_WRAP__
#endif // svtkDataArrayTupleRange_AOS_h

// SVTK-HeaderTest-Exclude: svtkDataArrayTupleRange_AOS.h

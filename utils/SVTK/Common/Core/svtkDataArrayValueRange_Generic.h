/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayValueRange_Generic.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * Generic implementation of value ranges and iterators, suitable for
 * svtkDataArray and all subclasses.
 */

#ifndef svtkDataArrayValueRange_Generic_h
#define svtkDataArrayValueRange_Generic_h

#include "svtkDataArrayAccessor.h"
#include "svtkDataArrayMeta.h"
#include "svtkIterator.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <tuple>
#include <type_traits>

#ifndef __SVTK_WRAP__

SVTK_ITER_OPTIMIZE_START

namespace svtk
{

namespace detail
{

// Forward decs for friends/args
template <typename ArrayType, ComponentIdType>
struct ValueReference;
template <typename ArrayType, ComponentIdType>
struct ConstValueReference;
template <typename ArrayType, ComponentIdType>
struct ValueIterator;
template <typename ArrayType, ComponentIdType>
struct ConstValueIterator;
template <typename ArrayType, ComponentIdType>
struct ValueRange;

//------------------------------------------------------------------------------
// Helper that converts ValueId <--> { TupleId, ComponentId }
// This class stores both representations. Profiling and assembly inspection
// show that ValueId is much more efficient for comparing Ids, while Tuple/Comp
// ids are much faster for looking up elements (especially when considering
// SOA arrays). The overhead of maintaining both is low, and this class is
// transparent enough that the compiler will produce efficient ASM with
// simple optimizations enabled.
template <ComponentIdType TupleSize>
struct IdStorage
{
  using NumCompsType = GenericTupleSize<TupleSize>;

  SVTK_ITER_INLINE
  IdStorage() noexcept
    : ValueId(0)
    , TupleId(0)
    , ComponentId(0)
  {
  }

  SVTK_ITER_INLINE
  IdStorage(ValueIdType valueId, NumCompsType numComps) noexcept
    : ValueId(valueId)
    , TupleId(static_cast<TupleIdType>(valueId) / static_cast<TupleIdType>(numComps.value))
    , ComponentId(static_cast<ComponentIdType>(valueId % static_cast<ValueIdType>(numComps.value)))
    , NumComps(numComps)
  {
  }

  SVTK_ITER_INLINE
  IdStorage(TupleIdType tupleId, ComponentIdType comp, NumCompsType numComps) noexcept
    : ValueId(tupleId* numComps.value + comp)
    , TupleId(tupleId)
    , ComponentId(comp)
    , NumComps(numComps)
  {
  }

  SVTK_ITER_INLINE
  IdStorage(
    ValueIdType valueId, TupleIdType tupleId, ComponentIdType comp, NumCompsType numComps) noexcept
    : ValueId(valueId)
    , TupleId(tupleId)
    , ComponentId(comp)
    , NumComps(numComps)
  {
  }

  template <typename ArrayType>
  SVTK_ITER_INLINE void DebugAsserts(ArrayType* array) const noexcept
  {
    (void)array;
    SVTK_ITER_ASSERT(array != nullptr, "Invalid array.");
    SVTK_ITER_ASSERT(this->ValueId == this->TupleId * this->GetTupleSize() + this->ComponentId,
      "Inconsistent internal state in IdStorage.");
    SVTK_ITER_ASSERT(this->GetTupleSize() > 0, "Invalid number of components.");
    SVTK_ITER_ASSERT(
      this->ValueId >= 0 && this->ValueId <= array->GetNumberOfValues(), "Invalid value id.");
    SVTK_ITER_ASSERT(this->GetTupleId() >= 0 && this->GetTupleId() <= array->GetNumberOfTuples(),
      "Invalid tuple id.");
    SVTK_ITER_ASSERT(this->GetComponentId() >= 0 &&
        (this->GetComponentId() < this->GetTupleSize() ||
          (this->GetComponentId() == this->GetTupleSize() &&
            this->GetTupleId() == array->GetNumberOfTuples())),
      "Invalid component id.");
    SVTK_ITER_ASSERT(this->GetValueId() >= 0 && this->GetValueId() <= array->GetNumberOfValues(),
      "Invalid value id.");
  }

  SVTK_ITER_INLINE
  IdStorage& operator++() noexcept // prefix
  {
    ++this->ValueId;
    ++this->ComponentId;
    if (this->ComponentId == this->GetTupleSize())
    {
      this->ComponentId = 0;
      ++this->TupleId;
    }
    return *this;
  }

  SVTK_ITER_INLINE
  IdStorage operator++(int) noexcept // postfix
  {
    auto v = this->ValueId++;
    auto t = this->TupleId;
    auto c = this->ComponentId++;
    if (this->ComponentId == this->GetTupleSize())
    {
      this->ComponentId = 0;
      ++this->TupleId;
    }
    return IdStorage{ v, t, c, this->NumComps };
  }

  friend SVTK_ITER_INLINE IdStorage operator+(const IdStorage& id, ValueIdType offset) noexcept
  {
    IdStorage res = id;
    res.AddOffset(offset);
    return res;
  }

  SVTK_ITER_INLINE
  IdStorage& operator--() noexcept // prefix
  {
    --this->ValueId;
    --this->ComponentId;
    if (this->ComponentId < 0)
    {
      this->ComponentId = this->GetTupleSize() - 1;
      --this->TupleId;
    }
    return *this;
  }

  SVTK_ITER_INLINE
  IdStorage operator--(int) noexcept // postfix
  {
    auto v = this->ValueId--;
    auto t = this->TupleId;
    auto c = this->ComponentId--;
    if (this->ComponentId < 0)
    {
      this->ComponentId = this->GetTupleSize() - 1;
      --this->TupleId;
    }
    return IdStorage{ v, t, c, this->NumComps };
  }

  SVTK_ITER_INLINE
  ValueIdType Convert(TupleIdType tuple, ComponentIdType comp) const noexcept
  {
    return static_cast<ValueIdType>(tuple) * this->NumComps.value + comp;
  }

  SVTK_ITER_INLINE
  std::pair<TupleIdType, ComponentIdType> Convert(ValueIdType value) const noexcept
  {
    return std::make_pair(static_cast<TupleIdType>(value / this->NumComps.value),
      static_cast<ComponentIdType>(value % this->NumComps.value));
  }

  SVTK_ITER_INLINE
  void AddOffset(ValueIdType offset) noexcept
  {
    this->ValueId += offset;
    std::tie(this->TupleId, this->ComponentId) = this->Convert(this->ValueId);
  }

  SVTK_ITER_INLINE
  ComponentIdType GetTupleSize() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  TupleIdType GetTupleId() const noexcept { return this->TupleId; }

  SVTK_ITER_INLINE
  ComponentIdType GetComponentId() const noexcept { return this->ComponentId; }

  SVTK_ITER_INLINE
  ValueIdType GetValueId() const noexcept { return this->ValueId; }

  friend SVTK_ITER_INLINE void swap(IdStorage& lhs, IdStorage& rhs) noexcept
  {
    using std::swap;
    swap(lhs.ValueId, rhs.ValueId);
    swap(lhs.TupleId, rhs.TupleId);
    swap(lhs.ComponentId, rhs.ComponentId);
  }

private:
  svtk::ValueIdType ValueId;
  svtk::TupleIdType TupleId;
  svtk::ComponentIdType ComponentId;
  NumCompsType NumComps;
};

//------------------------------------------------------------------------------
// Value reference
template <typename ArrayType, ComponentIdType TupleSize>
struct ConstValueReference
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayType>::value, "Invalid array type.");

  using IdStorageType = IdStorage<TupleSize>;
  using APIType = GetAPIType<ArrayType>;

public:
  SVTK_ITER_INLINE
  ConstValueReference() noexcept
    : Array{ nullptr }
    , Id{}
  {
  }

  SVTK_ITER_INLINE
  ConstValueReference(ArrayType* array, IdStorageType id) noexcept
    : Array{ array }
    , Id{ id }
  {
    this->Id.DebugAsserts(array);
  }

  SVTK_ITER_INLINE
  ConstValueReference(const ValueReference<ArrayType, TupleSize>& o)
    : Array{ o.Array }
    , Id{ o.Id }
  {
  }

  SVTK_ITER_INLINE
  ConstValueReference(const ConstValueReference& o) noexcept = default;

  SVTK_ITER_INLINE
  ConstValueReference(ConstValueReference&& o) noexcept = default;

  SVTK_ITER_INLINE
  ConstValueReference operator=(const ConstValueReference& o) noexcept
  {
    SVTK_ITER_ASSERT(!this->Array, "Const reference already initialized.");
    // Initialize the reference.
    this->Array = o.Array;
    this->Id = o.Id;
    return *this;
  }

  SVTK_ITER_INLINE
  ConstValueReference operator=(ConstValueReference&& o) noexcept
  {
    SVTK_ITER_ASSERT(!this->Array, "Const reference already initialized.");
    // Initialize the reference.
    this->Array = std::move(o.Array);
    this->Id = std::move(o.Id);
    return *this;
  }

  SVTK_ITER_INLINE
  operator APIType() const noexcept
  {
    SVTK_ITER_ASSUME(this->Id.GetTupleSize() > 0);
    SVTK_ITER_ASSUME(this->Array->GetNumberOfComponents() == this->Id.GetTupleSize());
    svtkDataArrayAccessor<ArrayType> acc{ this->Array };
    return acc.Get(this->Id.GetTupleId(), this->Id.GetComponentId());
  }

protected:
  mutable ArrayType* Array;
  IdStorageType Id;
};

//------------------------------------------------------------------------------
// Value reference
template <typename ArrayType, ComponentIdType TupleSize>
struct ValueReference
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayType>::value, "Invalid array type.");

  using APIType = GetAPIType<ArrayType>;
  using IdStorageType = IdStorage<TupleSize>;

public:
  using value_type = APIType;

  SVTK_ITER_INLINE
  ValueReference() noexcept
    : Array{ nullptr }
    , Id{}
  {
  }

  SVTK_ITER_INLINE
  ValueReference(ArrayType* array, IdStorageType id) noexcept
    : Array{ array }
    , Id{ id }
  {
    this->Id.DebugAsserts(this->Array);
  }

  SVTK_ITER_INLINE
  ValueReference(const ValueReference& o) noexcept = default;
  SVTK_ITER_INLINE
  ValueReference(ValueReference&& o) noexcept = default;

  SVTK_ITER_INLINE
  ValueReference operator=(const ValueReference& o) noexcept
  {
    if (this->Array)
    { // Already initialized. Assign the value, not the reference:
      return *this = static_cast<APIType>(o);
    }
    else
    { // Initialize the reference:
      this->Array = o.Array;
      this->Id = o.Id;
      return *this;
    }
  }

  SVTK_ITER_INLINE
  ValueReference operator=(ValueReference&& o) noexcept
  {
    if (this->Array)
    { // Already initialized. Assign the value, not the reference:
      return *this = static_cast<APIType>(o);
    }
    else
    { // Initialize the reference:
      this->Array = std::move(o.Array);
      this->Id = std::move(o.Id);
      return *this;
    }
  }

  template <typename OArray, ComponentIdType OSize>
  SVTK_ITER_INLINE ValueReference operator=(const ValueReference<OArray, OSize>& o) noexcept
  { // Always copy the value for different reference types:
    const APIType tmp = o;
    return *this = std::move(tmp);
  }

  SVTK_ITER_INLINE
  operator APIType() const noexcept
  {
    SVTK_ITER_ASSUME(this->Id.GetTupleSize() > 0);
    SVTK_ITER_ASSUME(this->Array->GetNumberOfComponents() == this->Id.GetTupleSize());
    svtkDataArrayAccessor<ArrayType> acc{ this->Array };
    return acc.Get(this->Id.GetTupleId(), this->Id.GetComponentId());
  }

  SVTK_ITER_INLINE
  ValueReference operator=(APIType val) noexcept
  {
    SVTK_ITER_ASSUME(this->Id.GetTupleSize() > 0);
    SVTK_ITER_ASSUME(this->Array->GetNumberOfComponents() == this->Id.GetTupleSize());
    svtkDataArrayAccessor<ArrayType> acc{ this->Array };
    acc.Set(this->Id.GetTupleId(), this->Id.GetComponentId(), val);
    return *this;
  }

  friend SVTK_ITER_INLINE void swap(ValueReference lhs, ValueReference rhs) noexcept
  { // Swap values, not references:
    APIType tmp = std::move(static_cast<APIType>(lhs));
    lhs = std::move(static_cast<APIType>(rhs));
    rhs = std::move(tmp);
  }

  template <typename OArray, ComponentIdType OSize>
  friend SVTK_ITER_INLINE void swap(ValueReference lhs, ValueReference<OArray, OSize> rhs) noexcept
  { // Swap values, not references:
    using OAPIType = typename ValueReference<OArray, OSize>::value_type;
    static_assert(
      std::is_same<APIType, OAPIType>::value, "Cannot swap components with different types.");

    APIType tmp = std::move(static_cast<APIType>(lhs));
    lhs = std::move(static_cast<APIType>(rhs));
    rhs = std::move(tmp);
  }

  friend SVTK_ITER_INLINE void swap(ValueReference lhs, APIType& rhs) noexcept
  {
    APIType tmp = std::move(static_cast<APIType>(lhs));
    lhs = std::move(rhs);
    rhs = std::move(tmp);
  }

  friend SVTK_ITER_INLINE void swap(APIType& lhs, ValueReference rhs) noexcept
  {
    APIType tmp = std::move(lhs);
    lhs = std::move(static_cast<APIType>(rhs));
    rhs = std::move(tmp);
  }

  SVTK_ITER_INLINE
  ValueReference operator++() noexcept // prefix
  {
    const APIType newVal = *this + 1;
    *this = newVal;
    return *this;
  }

  SVTK_ITER_INLINE
  APIType operator++(int) noexcept // postfix
  {
    const APIType retVal = *this;
    *this = *this + 1;
    return retVal;
  }

  SVTK_ITER_INLINE
  ValueReference operator--() noexcept // prefix
  {
    const APIType newVal = *this - 1;
    *this = newVal;
    return *this;
  }

  SVTK_ITER_INLINE
  APIType operator--(int) noexcept // postfix
  {
    const APIType retVal = *this;
    *this = *this - 1;
    return retVal;
  }

#define SVTK_REF_OP_OVERLOADS(Op, ImplOp)                                                           \
  friend SVTK_ITER_INLINE ValueReference operator Op(ValueReference lhs, APIType val) noexcept      \
  {                                                                                                \
    const APIType newVal = lhs ImplOp val;                                                         \
    lhs = newVal;                                                                                  \
    return lhs;                                                                                    \
  }                                                                                                \
  friend SVTK_ITER_INLINE ValueReference operator Op(ValueReference lhs, ValueReference val)        \
    noexcept                                                                                       \
  {                                                                                                \
    const APIType newVal = lhs ImplOp val;                                                         \
    lhs = newVal;                                                                                  \
    return lhs;                                                                                    \
  }                                                                                                \
  friend SVTK_ITER_INLINE APIType& operator Op(APIType& lhs, ValueReference val) noexcept           \
  {                                                                                                \
    const APIType newVal = lhs ImplOp val;                                                         \
    lhs = newVal;                                                                                  \
    return lhs;                                                                                    \
  }

  SVTK_REF_OP_OVERLOADS(+=, +)
  SVTK_REF_OP_OVERLOADS(-=, -)
  SVTK_REF_OP_OVERLOADS(*=, *)
  SVTK_REF_OP_OVERLOADS(/=, /)

#undef SVTK_REF_OP_OVERLOADS

  friend struct ConstValueReference<ArrayType, TupleSize>;
  friend struct ValueIterator<ArrayType, TupleSize>;

protected:
  void CopyReference(const ValueReference& o) noexcept
  {
    this->Array = o.Array;
    this->Id = o.Id;
  }

  mutable ArrayType* Array;
  IdStorageType Id;
};

//------------------------------------------------------------------------------
// Const value iterator
template <typename ArrayType, ComponentIdType TupleSize>
struct ConstValueIterator
  : public svtkIterator<std::random_access_iterator_tag, GetAPIType<ArrayType>, ValueIdType,
      // expected types don't have members, no op->().
      void, ConstValueReference<ArrayType, TupleSize> >
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayType>::value, "Invalid array type.");

  using APIType = GetAPIType<ArrayType>;
  using IdStorageType = IdStorage<TupleSize>;
  using Superclass = svtkIterator<std::random_access_iterator_tag, APIType, ValueIdType, void,
    ConstValueReference<ArrayType, TupleSize> >;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  SVTK_ITER_INLINE
  ConstValueIterator() noexcept
    : Array(nullptr)
    , Id()
  {
  }

  SVTK_ITER_INLINE
  ConstValueIterator(ArrayType* array, IdStorageType id) noexcept
    : Array(array)
    , Id(id)
  {
    this->Id.DebugAsserts(this->Array);
  }

  SVTK_ITER_INLINE
  ConstValueIterator(const ValueIterator<ArrayType, TupleSize>& o) noexcept
    : Array{ o.GetArray() }
    , Id{ o.GetId() }
  {
  }

  SVTK_ITER_INLINE
  ConstValueIterator(const ConstValueIterator& o) noexcept = default;
  SVTK_ITER_INLINE
  ConstValueIterator& operator=(const ConstValueIterator& o) noexcept = default;

  SVTK_ITER_INLINE
  ConstValueIterator& operator++() noexcept // prefix
  {
    ++this->Id;
    this->Id.DebugAsserts(this->Array);
    return *this;
  }

  SVTK_ITER_INLINE
  ConstValueIterator operator++(int) noexcept // postfix
  {
    auto ret = this->Id++;
    this->Id.DebugAsserts(this->Array);
    return ConstValueIterator{ this->Array, ret };
  }

  SVTK_ITER_INLINE
  ConstValueIterator& operator--() noexcept // prefix
  {
    --this->Id;
    this->Id.DebugAsserts(this->Array);
    return *this;
  }

  SVTK_ITER_INLINE
  ConstValueIterator operator--(int) noexcept // postfix
  {
    auto ret = this->Id--;
    this->Id.DebugAsserts(this->Array);
    return ConstValueIterator{ this->Array, ret };
  }

  SVTK_ITER_INLINE
  reference operator[](difference_type i) const noexcept
  {
    return reference{ this->Array, this->Id + i };
  }

  SVTK_ITER_INLINE
  reference operator*() const noexcept { return reference{ this->Array, this->Id }; }

  // Using GetValueType here makes iteration 50% faster by reducing comparisons
  // and jumps (instead of comparing std::tie(tupleId, compId)).
#define SVTK_TMP_MAKE_OPERATOR(OP)                                                                  \
  friend SVTK_ITER_INLINE bool operator OP(                                                         \
    const ConstValueIterator& lhs, const ConstValueIterator& rhs) noexcept                         \
  {                                                                                                \
    SVTK_ITER_ASSERT(lhs.Array == rhs.Array, "Mismatched arrays in iterator comparison.");          \
    return lhs.Id.GetValueId() OP rhs.Id.GetValueId();                                             \
  }

  SVTK_TMP_MAKE_OPERATOR(==)
  SVTK_TMP_MAKE_OPERATOR(!=)
  SVTK_TMP_MAKE_OPERATOR(<)
  SVTK_TMP_MAKE_OPERATOR(>)
  SVTK_TMP_MAKE_OPERATOR(<=)
  SVTK_TMP_MAKE_OPERATOR(>=)

#undef SVTK_TMP_MAKE_OPERATOR

  SVTK_ITER_INLINE
  ConstValueIterator& operator+=(difference_type offset) noexcept
  {
    this->Id.AddOffset(offset);
    this->Id.DebugAsserts(this->Array);
    return *this;
  }

  friend SVTK_ITER_INLINE ConstValueIterator operator+(
    const ConstValueIterator& it, difference_type offset) noexcept
  {
    return ConstValueIterator{ it.Array, it.Id + offset };
  }

  friend SVTK_ITER_INLINE ConstValueIterator operator+(
    difference_type offset, const ConstValueIterator& it) noexcept
  {
    return ConstValueIterator{ it.Array, it.Id + offset };
  }

  SVTK_ITER_INLINE
  ConstValueIterator& operator-=(difference_type offset) noexcept
  {
    this->Id.AddOffset(-offset);
    this->Id.DebugAsserts(this->Array);
    return *this;
  }

  friend SVTK_ITER_INLINE ConstValueIterator operator-(
    const ConstValueIterator& it, difference_type offset) noexcept
  {
    return ConstValueIterator{ it.Array, it.Id + (-offset) };
  }

  friend SVTK_ITER_INLINE difference_type operator-(
    const ConstValueIterator& it1, const ConstValueIterator& it2) noexcept
  {
    SVTK_ITER_ASSERT(it1.Array == it2.Array, "Cannot do math with iterators from different arrays.");
    return it1.Id.GetValueId() - it2.Id.GetValueId();
  }

  friend SVTK_ITER_INLINE void swap(ConstValueIterator& lhs, ConstValueIterator& rhs) noexcept
  {
    // Different arrays may use different iterator implementations.
    SVTK_ITER_ASSERT(lhs.Array == rhs.Array, "Cannot swap iterators from different arrays.");

    using std::swap;
    swap(lhs.Id, rhs.Id);
  }

private:
  mutable ArrayType* Array;
  IdStorageType Id;
};

//------------------------------------------------------------------------------
// Component iterator
template <typename ArrayType, ComponentIdType TupleSize>
struct ValueIterator
  : public svtkIterator<std::random_access_iterator_tag, svtk::GetAPIType<ArrayType>, ValueIdType,
      ValueReference<ArrayType, TupleSize>, ValueReference<ArrayType, TupleSize> >
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayType>::value, "Invalid array type.");

  using APIType = GetAPIType<ArrayType>;
  using IdStorageType = IdStorage<TupleSize>;
  using Superclass = svtkIterator<std::random_access_iterator_tag, GetAPIType<ArrayType>,
    ValueIdType, ValueReference<ArrayType, TupleSize>, ValueReference<ArrayType, TupleSize> >;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  SVTK_ITER_INLINE
  ValueIterator() noexcept = default;

  SVTK_ITER_INLINE
  ValueIterator(ArrayType* array, IdStorageType id) noexcept : Ref{ array, id }
  {
    this->DebugIdAsserts();
  }

  SVTK_ITER_INLINE
  ValueIterator(const ValueIterator& o) noexcept = default;

  SVTK_ITER_INLINE
  ValueIterator& operator=(const ValueIterator& o) noexcept
  {
    this->Ref.CopyReference(o.Ref);
    this->DebugIdAsserts();
    return *this;
  }

  SVTK_ITER_INLINE
  ValueIterator& operator++() noexcept // prefix
  {
    ++this->Ref.Id;
    this->DebugIdAsserts();
    return *this;
  }

  SVTK_ITER_INLINE
  ValueIterator operator++(int) noexcept // postfix
  {
    auto ret = this->Ref.Id++;
    this->DebugIdAsserts();
    return ValueIterator{ this->Ref.Array, ret };
  }

  SVTK_ITER_INLINE
  ValueIterator& operator--() noexcept // prefix
  {
    --this->Ref.Id;
    this->DebugIdAsserts();
    return *this;
  }

  SVTK_ITER_INLINE
  ValueIterator operator--(int) noexcept // postfix
  {
    auto ret = this->Ref.Id--;
    this->DebugIdAsserts();
    return ValueIterator{ this->Ref.Array, ret };
  }

  SVTK_ITER_INLINE
  reference operator[](difference_type i) const noexcept
  {
    return reference{ this->Ref.Array, this->Ref.Id + i };
  }

  SVTK_ITER_INLINE
  reference operator*() const noexcept { return this->Ref; }

  SVTK_ITER_INLINE
  const pointer& operator->() const noexcept { return this->Ref; }

#define SVTK_TMP_MAKE_OPERATOR(OP)                                                                  \
  friend SVTK_ITER_INLINE bool operator OP(const ValueIterator& lhs, const ValueIterator& rhs)      \
    noexcept                                                                                       \
  {                                                                                                \
    SVTK_ITER_ASSERT(                                                                               \
      lhs.GetArray() == rhs.GetArray(), "Mismatched arrays in iterator comparison.");              \
    return lhs.GetId().GetValueId() OP rhs.GetId().GetValueId();                                   \
  }

  SVTK_TMP_MAKE_OPERATOR(==)
  SVTK_TMP_MAKE_OPERATOR(!=)
  SVTK_TMP_MAKE_OPERATOR(<)
  SVTK_TMP_MAKE_OPERATOR(>)
  SVTK_TMP_MAKE_OPERATOR(<=)
  SVTK_TMP_MAKE_OPERATOR(>=)

#undef SVTK_TMP_MAKE_OPERATOR

  SVTK_ITER_INLINE
  ValueIterator& operator+=(difference_type offset) noexcept
  {
    this->Ref.Id.AddOffset(offset);
    this->DebugIdAsserts();
    return *this;
  }

  friend SVTK_ITER_INLINE ValueIterator operator+(
    const ValueIterator& it, difference_type offset) noexcept
  {
    return ValueIterator{ it.GetArray(), it.GetId() + offset };
  }

  friend SVTK_ITER_INLINE ValueIterator operator+(
    difference_type offset, const ValueIterator& it) noexcept
  {
    return ValueIterator{ it.GetArray(), it.GetId() + offset };
  }

  SVTK_ITER_INLINE
  ValueIterator& operator-=(difference_type offset) noexcept
  {
    this->Ref.Id.AddOffset(-offset);
    this->Ref.Id.DebugAsserts(this->Ref.Array);
    return *this;
  }

  friend SVTK_ITER_INLINE ValueIterator operator-(
    const ValueIterator& it, difference_type offset) noexcept
  {
    return ValueIterator{ it.GetArray(), it.GetId() + (-offset) };
  }

  friend SVTK_ITER_INLINE difference_type operator-(
    const ValueIterator& it1, const ValueIterator& it2) noexcept
  {
    SVTK_ITER_ASSERT(
      it1.Ref.Array == it2.Ref.Array, "Cannot do math with iterators from different arrays.");
    return it1.GetId().GetValueId() - it2.GetId().GetValueId();
  }

  friend SVTK_ITER_INLINE void swap(ValueIterator& lhs, ValueIterator& rhs) noexcept
  {
    // Different arrays may use different iterator implementations.
    SVTK_ITER_ASSERT(
      lhs.GetArray() == rhs.GetArray(), "Cannot swap iterators from different arrays.");

    using std::swap;
    swap(lhs.GetId(), rhs.GetId());
  }

  friend struct ConstValueIterator<ArrayType, TupleSize>;

protected:
  SVTK_ITER_INLINE
  void DebugIdAsserts() const { this->Ref.Id.DebugAsserts(this->Ref.Array); }

  // Needed for access from friend functions. We could just store the array
  // and ID here instead of the ref, but meh.
  ArrayType* GetArray() const noexcept { return this->Ref.Array; }
  IdStorageType& GetId() noexcept { return this->Ref.Id; }
  const IdStorageType& GetId() const noexcept { return this->Ref.Id; }

  ValueReference<ArrayType, TupleSize> Ref;
};

//------------------------------------------------------------------------------
// ValueRange
template <typename ArrayTypeT, ComponentIdType TupleSize>
struct ValueRange
{
private:
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(IsVtkDataArray<ArrayTypeT>::value, "Invalid array type.");

  using IdStorageType = IdStorage<TupleSize>;
  using NumCompsType = GenericTupleSize<TupleSize>;

public:
  using ArrayType = ArrayTypeT;
  using ValueType = GetAPIType<ArrayTypeT>;

  using IteratorType = ValueIterator<ArrayType, TupleSize>;
  using ConstIteratorType = ConstValueIterator<ArrayType, TupleSize>;
  using ReferenceType = ValueReference<ArrayType, TupleSize>;
  using ConstReferenceType = ConstValueReference<ArrayType, TupleSize>;

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
    , BeginValue(beginValue, this->NumComps)
    , EndValue(endValue, this->NumComps)
  {
    assert(this->Array);
    assert(beginValue >= 0 && beginValue <= endValue);
    assert(endValue >= 0 && endValue <= this->Array->GetNumberOfValues());
  }

  SVTK_ITER_INLINE
  ValueRange GetSubRange(ValueIdType beginValue = 0, ValueIdType endValue = -1) const noexcept
  {
    const ValueIdType realBegin = this->BeginValue.GetValueId() + beginValue;
    const ValueIdType realEnd =
      endValue >= 0 ? this->BeginValue.GetValueId() + endValue : this->EndValue.GetValueId();

    return ValueRange{ this->Array, realBegin, realEnd };
  }

  SVTK_ITER_INLINE
  ArrayType* GetArray() const noexcept { return this->Array; }
  SVTK_ITER_INLINE
  ComponentIdType GetTupleSize() const noexcept { return this->NumComps.value; }

  SVTK_ITER_INLINE
  ValueIdType GetBeginValueId() const noexcept { return this->BeginValue.GetValueId(); }

  SVTK_ITER_INLINE
  ValueIdType GetEndValueId() const noexcept { return this->EndValue.GetValueId(); }

  SVTK_ITER_INLINE
  size_type size() const noexcept
  {
    return this->EndValue.GetValueId() - this->BeginValue.GetValueId();
  }

  SVTK_ITER_INLINE
  iterator begin() noexcept { return this->NewIterator(this->BeginValue); }
  SVTK_ITER_INLINE
  iterator end() noexcept { return this->NewIterator(this->EndValue); }

  SVTK_ITER_INLINE
  const_iterator begin() const noexcept { return this->NewConstIterator(this->BeginValue); }
  SVTK_ITER_INLINE
  const_iterator end() const noexcept { return this->NewConstIterator(this->EndValue); }

  SVTK_ITER_INLINE
  const_iterator cbegin() const noexcept { return this->NewConstIterator(this->BeginValue); }
  SVTK_ITER_INLINE
  const_iterator cend() const noexcept { return this->NewConstIterator(this->EndValue); }

  SVTK_ITER_INLINE
  reference operator[](size_type i) noexcept
  {
    return reference{ this->Array, this->BeginValue + i };
  }
  SVTK_ITER_INLINE
  const_reference operator[](size_type i) const noexcept
  {
    return const_reference{ this->Array, this->BeginValue + i };
  }

private:
  SVTK_ITER_INLINE
  iterator NewIterator(IdStorageType id) const noexcept { return iterator{ this->Array, id }; }

  SVTK_ITER_INLINE
  const_iterator NewConstIterator(IdStorageType id) const noexcept
  {
    return const_iterator{ this->Array, id };
  }

  mutable ArrayType* Array{ nullptr };
  NumCompsType NumComps{};
  IdStorageType BeginValue{};
  IdStorageType EndValue{};
};

// Unimplemented, only used inside decltype in SelectValueRange:
template <typename ArrayType, ComponentIdType TupleSize>
ValueRange<ArrayType, TupleSize> DeclareValueRangeSpecialization(svtkDataArray*);

} // end namespace detail
} // end namespace svtk

SVTK_ITER_OPTIMIZE_END

#endif // __SVTK_WRAP__
#endif // svtkDataArrayValueRange_Generic_h

// SVTK-HeaderTest-Exclude: svtkDataArrayValueRange_Generic.h

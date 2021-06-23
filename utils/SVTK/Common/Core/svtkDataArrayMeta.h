/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayMeta.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkDataArrayMeta_h
#define svtkDataArrayMeta_h

#include "svtkAssume.h"
#include "svtkConfigure.h"
#include "svtkDataArray.h"
#include "svtkMeta.h"
#include "svtkSetGet.h"
#include "svtkType.h"

#include <type_traits>
#include <utility>

/**
 * @file svtkDataArrayMeta.h
 * This file contains a variety of metaprogramming constructs for working
 * with svtkDataArrays.
 */

// When enabled, extra debugging checks are enabled for the iterators.
// Specifically:
// - Specializations are disabled (All code uses the generic implementation).
// - Additional assertions are inserted to ensure correct runtime usage.
// - Performance-related annotations (e.g. force inlining) are disabled.
#if defined(SVTK_DEBUG_RANGE_ITERATORS)
#define SVTK_ITER_ASSERT(x, msg) assert((x) && msg)
#else
#define SVTK_ITER_ASSERT(x, msg)
#endif

#if defined(SVTK_ALWAYS_OPTIMIZE_ARRAY_ITERATORS) && !defined(SVTK_DEBUG_RANGE_ITERATORS)
#define SVTK_ITER_INLINE SVTK_ALWAYS_INLINE
#define SVTK_ITER_ASSUME SVTK_ASSUME_NO_ASSERT
#define SVTK_ITER_OPTIMIZE_START SVTK_ALWAYS_OPTIMIZE_START
#define SVTK_ITER_OPTIMIZE_END SVTK_ALWAYS_OPTIMIZE_START
#else
#define SVTK_ITER_INLINE inline
#define SVTK_ITER_ASSUME SVTK_ASSUME
#define SVTK_ITER_OPTIMIZE_START
#define SVTK_ITER_OPTIMIZE_END
#endif

SVTK_ITER_OPTIMIZE_START

// For IsAOSDataArray:
template <typename ValueType>
class svtkAOSDataArrayTemplate;

namespace svtk
{

// Typedef for data array indices:
using ComponentIdType = int;
using TupleIdType = svtkIdType;
using ValueIdType = svtkIdType;

namespace detail
{

//------------------------------------------------------------------------------
// Used by ranges/iterators when tuple size is unknown at compile time
static constexpr ComponentIdType DynamicTupleSize = 0;

//------------------------------------------------------------------------------
// Detect data array value types
template <typename T>
struct IsVtkDataArray : std::is_base_of<svtkDataArray, T>
{
};

template <typename T>
using EnableIfVtkDataArray = typename std::enable_if<IsVtkDataArray<T>::value>::type;

//------------------------------------------------------------------------------
// If a value is a valid tuple size
template <ComponentIdType Size>
struct IsValidTupleSize : std::integral_constant<bool, (Size > 0 || Size == DynamicTupleSize)>
{
};

template <ComponentIdType TupleSize>
using EnableIfValidTupleSize = typename std::enable_if<IsValidTupleSize<TupleSize>::value>::type;

//------------------------------------------------------------------------------
// If a value is a non-dynamic tuple size
template <ComponentIdType Size>
struct IsStaticTupleSize : std::integral_constant<bool, (Size > 0)>
{
};

template <ComponentIdType TupleSize>
using EnableIfStaticTupleSize = typename std::enable_if<IsStaticTupleSize<TupleSize>::value>::type;

//------------------------------------------------------------------------------
// If two values are valid non-dynamic tuple sizes:
template <ComponentIdType S1, ComponentIdType S2>
struct AreStaticTupleSizes
  : std::integral_constant<bool, (IsStaticTupleSize<S1>::value && IsStaticTupleSize<S2>::value)>
{
};

template <ComponentIdType S1, ComponentIdType S2, typename T = void>
using EnableIfStaticTupleSizes =
  typename std::enable_if<AreStaticTupleSizes<S1, S2>::value, T>::type;

//------------------------------------------------------------------------------
// If either of the tuple sizes is not statically defined
template <ComponentIdType S1, ComponentIdType S2>
struct IsEitherTupleSizeDynamic
  : std::integral_constant<bool, (!IsStaticTupleSize<S1>::value || !IsStaticTupleSize<S2>::value)>
{
};

template <ComponentIdType S1, ComponentIdType S2, typename T = void>
using EnableIfEitherTupleSizeIsDynamic =
  typename std::enable_if<IsEitherTupleSizeDynamic<S1, S2>::value, T>::type;

//------------------------------------------------------------------------------
// Helper that switches between a storageless integral constant for known
// sizes, and a runtime variable for variable sizes.
template <ComponentIdType TupleSize>
struct GenericTupleSize : public std::integral_constant<ComponentIdType, TupleSize>
{
  static_assert(IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");

private:
  using Superclass = std::integral_constant<ComponentIdType, TupleSize>;

public:
  // Need to construct from array for specialization.
  using Superclass::Superclass;
  SVTK_ITER_INLINE GenericTupleSize() noexcept = default;
  SVTK_ITER_INLINE GenericTupleSize(svtkDataArray*) noexcept {}
};

// Specialize for dynamic types, mimicking integral_constant API:
template <>
struct GenericTupleSize<DynamicTupleSize>
{
  using value_type = ComponentIdType;

  SVTK_ITER_INLINE GenericTupleSize() noexcept : value(0) {}
  SVTK_ITER_INLINE explicit GenericTupleSize(svtkDataArray* array)
    : value(array->GetNumberOfComponents())
  {
  }

  SVTK_ITER_INLINE operator value_type() const noexcept { return value; }
  SVTK_ITER_INLINE value_type operator()() const noexcept { return value; }

  ComponentIdType value;
};

template <typename ArrayType>
struct GetAPITypeImpl
{
  using APIType = typename ArrayType::ValueType;
};
template <>
struct GetAPITypeImpl<svtkDataArray>
{
  using APIType = double;
};

} // end namespace detail

//------------------------------------------------------------------------------
// Typedef for double if svtkDataArray, or the array's ValueType for subclasses.
template <typename ArrayType, typename = detail::EnableIfVtkDataArray<ArrayType> >
using GetAPIType = typename detail::GetAPITypeImpl<ArrayType>::APIType;

//------------------------------------------------------------------------------
namespace detail
{

template <typename ArrayType>
struct IsAOSDataArrayImpl
{
  using APIType = GetAPIType<ArrayType>;
  static constexpr bool value = std::is_base_of<svtkAOSDataArrayTemplate<APIType>, ArrayType>::value;
};

} // end namespace detail

//------------------------------------------------------------------------------
// True if ArrayType inherits some specialization of svtkAOSDataArrayTemplate
template <typename ArrayType>
using IsAOSDataArray = std::integral_constant<bool, detail::IsAOSDataArrayImpl<ArrayType>::value>;

} // end namespace svtk

SVTK_ITER_OPTIMIZE_END

#endif // svtkDataArrayMeta_h

// SVTK-HeaderTest-Exclude: svtkDataArrayMeta.h
